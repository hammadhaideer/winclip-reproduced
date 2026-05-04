"""WinCLIP scorer: text-image anomaly classification + multi-scale window segmentation.

Implements WinCLIP (zero-shot) and WinCLIP+ (few-normal-shot) following
Jeong et al., CVPR 2023.

The multi-scale window aggregation logic in `compute_window_score_map` mirrors
the protocol from the WinCLIP paper (Section 3.2) and the reference open-source
implementation in Intel's anomalib library
(https://github.com/openvinotoolkit/anomalib, MIT licensed).
This repository's contribution is the surrounding reproduction infrastructure:
dataset loading, evaluation pipeline, configs, and integration.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .prompts import build_prompt_ensemble
from .clip_backbone import (
    load_clip_backbone,
    encode_text,
    encode_image_with_tokens,
    compute_text_anchors,
    text_image_anomaly_score,
)


class WinCLIP:
    def __init__(
        self,
        model_name: str = "ViT-B-16-plus-240",
        pretrained: str = "laion400m_e32",
        scales: List[int] = (2, 3),
        device: str = "cuda",
    ):
        self.device = device
        self.scales = list(scales)

        self.model, self.tokenizer, self.preprocess = load_clip_backbone(
            model_name, pretrained, device,
        )

        self.normal_anchor: Optional[torch.Tensor] = None
        self.anomaly_anchor: Optional[torch.Tensor] = None
        self.reference_patch_feats: Optional[torch.Tensor] = None
        self.object_name: Optional[str] = None

    def setup_text(self, object_name: str) -> None:
        self.object_name = object_name
        prompts = build_prompt_ensemble(object_name)

        text_normal = encode_text(self.model, self.tokenizer, prompts["normal"], self.device)
        text_anomaly = encode_text(self.model, self.tokenizer, prompts["anomaly"], self.device)

        self.normal_anchor, self.anomaly_anchor = compute_text_anchors(
            text_normal, text_anomaly,
        )

    def setup_reference(self, reference_images: torch.Tensor) -> None:
        _, patch_feats = encode_image_with_tokens(self.model, reference_images, self.device)
        self.reference_patch_feats = patch_feats.reshape(-1, patch_feats.shape[-1])

    def _patch_grid_size(self, n_patches: int) -> int:
        side = int(round(n_patches ** 0.5))
        assert side * side == n_patches, f"Non-square patch grid: {n_patches}"
        return side

    def compute_window_score_map(
        self,
        patch_feats: torch.Tensor,
    ) -> torch.Tensor:
        b, n, d = patch_feats.shape
        side = self._patch_grid_size(n)
        feat_grid = patch_feats.reshape(b, side, side, d)

        score_maps = []
        for scale in self.scales:
            ws = scale
            unfolded = feat_grid.unfold(1, ws, 1).unfold(2, ws, 1)
            unfolded = unfolded.permute(0, 1, 2, 4, 5, 3).contiguous()
            window_feats = unfolded.reshape(b, -1, ws * ws, d).mean(dim=2)
            window_feats = F.normalize(window_feats, dim=-1)

            sim_normal = (window_feats @ self.normal_anchor.T) * 100.0
            sim_anomaly = (window_feats @ self.anomaly_anchor.T) * 100.0
            logits
