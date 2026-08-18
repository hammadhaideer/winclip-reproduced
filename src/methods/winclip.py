"""Minimal working WinCLIP scorer for reproduction runs."""

from typing import List, Optional

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
        self.normal_anchor, self.anomaly_anchor = compute_text_anchors(text_normal, text_anomaly)

    def setup_reference(self, reference_images: torch.Tensor) -> None:
        _, patch_feats = encode_image_with_tokens(self.model, reference_images, self.device)
        self.reference_patch_feats = patch_feats.reshape(-1, patch_feats.shape[-1])

    def _patch_grid_size(self, n_patches: int) -> int:
        side = int(round(n_patches ** 0.5))
        assert side * side == n_patches, f"Non-square patch grid: {n_patches}"
        return side

    def compute_patch_score_map(self, patch_feats: torch.Tensor) -> torch.Tensor:
        sim_normal = (patch_feats @ self.normal_anchor.T) * 100.0
        sim_anomaly = (patch_feats @ self.anomaly_anchor.T) * 100.0
        logits = torch.cat([sim_normal, sim_anomaly], dim=-1)
        probs = F.softmax(logits, dim=-1)
        patch_scores = probs[..., 1]
        side = self._patch_grid_size(patch_scores.shape[1])
        return patch_scores.reshape(patch_scores.shape[0], 1, side, side)

    def compute_window_score_map(self, patch_feats: torch.Tensor) -> torch.Tensor:
        b, n, d = patch_feats.shape
        side = self._patch_grid_size(n)
        feat_grid = patch_feats.reshape(b, side, side, d)

        maps = []
        for ws in self.scales:
            unfolded = feat_grid.unfold(1, ws, 1).unfold(2, ws, 1)
            unfolded = unfolded.permute(0, 1, 2, 4, 5, 3).contiguous()
            window_feats = unfolded.reshape(b, -1, ws * ws, d).mean(dim=2)
            window_feats = F.normalize(window_feats, dim=-1)

            sim_normal = (window_feats @ self.normal_anchor.T) * 100.0
            sim_anomaly = (window_feats @ self.anomaly_anchor.T) * 100.0
            logits = torch.cat([sim_normal, sim_anomaly], dim=-1)
            probs = F.softmax(logits, dim=-1)[..., 1]

            h = side - ws + 1
            window_map = probs.reshape(b, 1, h, h)
            window_map = F.interpolate(window_map, size=(side, side), mode="bilinear", align_corners=False)
            maps.append(window_map)

        if not maps:
            return self.compute_patch_score_map(patch_feats)
        maps.append(self.compute_patch_score_map(patch_feats))
        return torch.stack(maps, dim=0).mean(dim=0)

    def score_image(self, images: torch.Tensor, target_size: int = 240):
        if self.normal_anchor is None or self.anomaly_anchor is None:
            raise RuntimeError("Call setup_text() before scoring images.")

        cls_feat, patch_feats = encode_image_with_tokens(self.model, images, self.device)
        image_score = text_image_anomaly_score(cls_feat, self.normal_anchor, self.anomaly_anchor)

        score_map = self.compute_window_score_map(patch_feats)
        score_map = F.interpolate(score_map, size=(target_size, target_size), mode="bilinear", align_corners=False)
        score_map = score_map[:, 0]

        if self.reference_patch_feats is not None:
            # Lightweight WinCLIP+ style reference association.
            b, n, d = patch_feats.shape
            ref_sim = patch_feats.reshape(-1, d) @ self.reference_patch_feats.T
            ref_dist = 1.0 - ref_sim.max(dim=1).values
            ref_map = ref_dist.reshape(b, 1, self._patch_grid_size(n), self._patch_grid_size(n))
            ref_map = F.interpolate(ref_map, size=(target_size, target_size), mode="bilinear", align_corners=False)[:, 0]
            score_map = 0.5 * score_map + 0.5 * ref_map
            image_score = 0.5 * image_score + 0.5 * ref_dist.reshape(b, n).max(dim=1).values

        return image_score.detach().cpu(), score_map.detach().cpu()
