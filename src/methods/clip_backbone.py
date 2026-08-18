"""CLIP backbone loading and inference utilities for WinCLIP.

Wraps `open_clip` to provide a consistent interface for loading the ViT-B/16+ model
that WinCLIP uses, encoding images and text, and exposing intermediate
patch tokens for window-based aggregation.

Reference:
    Jeong et al., WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation,
    CVPR 2023. The paper uses ViT-B/16+ at 240x240 input resolution with the
    LAION-400M pretrained weights.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def load_clip_backbone(
    model_name: str = "ViT-B-16-plus-240",
    pretrained: str = "laion400m_e32",
    device: str = "cuda",
):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer, preprocess


@torch.no_grad()
def encode_text(
    model,
    tokenizer,
    prompts: List[str],
    device: str = "cuda",
    batch_size: int = 64,
) -> torch.Tensor:
    all_features = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        tokens = tokenizer(batch).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
        all_features.append(feats)
    return torch.cat(all_features, dim=0)


@torch.no_grad()
def encode_image_with_tokens(
    model,
    images: torch.Tensor,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = images.to(device)
    visual = model.visual

    x = visual.conv1(images)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device,
    )
    x = torch.cat([cls_token, x], dim=1)
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x)

    if visual.proj is not None:
        x = x @ visual.proj

    cls_feat = F.normalize(x[:, 0, :], dim=-1)
    patch_feats = F.normalize(x[:, 1:, :], dim=-1)
    return cls_feat, patch_feats


def compute_text_anchors(
    text_features_normal: torch.Tensor,
    text_features_anomaly: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normal_anchor = F.normalize(text_features_normal.mean(dim=0, keepdim=True), dim=-1)
    anomaly_anchor = F.normalize(text_features_anomaly.mean(dim=0, keepdim=True), dim=-1)
    return normal_anchor, anomaly_anchor


def text_image_anomaly_score(
    image_feat: torch.Tensor,
    normal_anchor: torch.Tensor,
    anomaly_anchor: torch.Tensor,
    temperature: float = 100.0,
) -> torch.Tensor:
    sim_normal = (image_feat @ normal_anchor.T) * temperature
    sim_anomaly = (image_feat @ anomaly_anchor.T) * temperature
    logits = torch.cat([sim_normal, sim_anomaly], dim=-1)
    probs = F.softmax(logits, dim=-1)
    return probs[:, 1]
