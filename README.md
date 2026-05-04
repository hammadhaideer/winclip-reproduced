# winclip-reproduced

> Clean reproduction of **WinCLIP** (Jeong et al., CVPR 2023) on MVTec-AD and VisA — zero-shot and few-normal-shot anomaly classification and segmentation with CLIP.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Paper: [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814) — Jeong et al., CVPR 2023

## What this is

End-to-end reproduction of WinCLIP — the first strong CLIP-based zero-shot anomaly detector — on MVTec-AD and VisA. Implements the compositional prompt ensemble, multi-scale window aggregation, and the few-normal-shot extension WinCLIP+.

This is the second repo in a series of visual anomaly detection reproductions building toward UniVAD (CVPR 2025), my main thesis baseline. WinCLIP is one of UniVAD's six comparison methods in Table 1.

## Status

🚧 In active development. Code lands over the coming week, results follow once GPU access returns.

## Why WinCLIP

WinCLIP shifted visual anomaly detection from "train a custom model per object" to "use a pretrained vision-language model with the right prompts." It's the first method on UniVAD's comparison list that uses a foundation model — making it conceptually the bridge between PatchCore-era methods (frozen ImageNet backbones) and the current foundation-model era of AD.

| | Zero-shot Image-AUROC | 1-shot Image-AUROC |
|---|---|---|
| WinCLIP | 91.8 | — |
| WinCLIP+ | — | 93.1 |
| PatchCore (1-shot, baseline) | — | 83.4 |

## Goal

Match the paper's reported numbers within ±0.5 points:

| Setting | Dataset | Image-AUROC | Pixel-AUROC |
|---|---|---|---|
| Zero-shot | MVTec-AD | 91.8 → TBD | 85.1 → TBD |
| Zero-shot | VisA | 78.1 → TBD | 79.6 → TBD |
| 1-normal-shot | MVTec-AD | 93.1 → TBD | 95.2 → TBD |
| 1-normal-shot | VisA | 83.8 → TBD | 96.4 → TBD |

## Roadmap

- [ ] Repo scaffold, configs, dataset loader (reused from patchcore-reproduced)
- [ ] CLIP ViT-B/16+ backbone integration
- [ ] Compositional prompt ensemble (state words × prompt templates)
- [ ] Multi-scale window aggregation (window/patch/image-level)
- [ ] Zero-shot anomaly classification and segmentation
- [ ] WinCLIP+ few-normal-shot extension with reference association
- [ ] Empirical reproduction on MVTec-AD and VisA
- [ ] Walkthrough notebook with prompt-ensemble visualization
- [ ] Medium walkthrough post

## Reproduction series

Part of a broader series reproducing UniVAD's full comparison set:

- [x] [patchcore-reproduced](https://github.com/hammadhaideer/patchcore-reproduced) — PatchCore (CVPR 2022)
- [ ] **winclip-reproduced** — WinCLIP (CVPR 2023) ← *this repo*
- [ ] uniad-reproduced — UniAD (NeurIPS 2022)
- [ ] anomalygpt-reproduced — AnomalyGPT (AAAI 2024)
- [ ] comad-reproduced — ComAD (PR 2024)
- [ ] medclip-reproduced — MedCLIP (EMNLP 2022)
- [ ] univad-reproduced — UniVAD (CVPR 2025) ← *target*

## References

- Jeong et al., *WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation*, CVPR 2023
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021
- Bergmann et al., *MVTec-AD*, CVPR 2019
- Zou et al., *VisA*, ECCV 2022

## License

MIT
