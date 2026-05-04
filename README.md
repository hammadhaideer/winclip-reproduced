# efficientad-reproduced

> Clean reproduction of **EfficientAD** (Batzner et al., WACV 2024) on MVTec-AD with image-level AUROC, pixel-level AUROC, and pixel-level AUPRO.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Paper: [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535) - Batzner et al., WACV 2024

## What this is

End-to-end reproduction of EfficientAD using a student-teacher distillation pair plus an autoencoder for global anomaly detection. The method achieves sub-millisecond inference on a single GPU while matching the AUROC of much heavier methods like PatchCore.

This is the second repo in a series of visual anomaly detection reproductions I'm building during May–June 2026, following [patchcore-reproduced](https://github.com/hammadhaideer/patchcore-reproduced).

## Status

In active development. Code lands over the coming week, results follow once GPU access returns.

## Why EfficientAD

PatchCore's reported numbers are excellent, but its memory bank + nearest-neighbour scoring is too slow for real-time industrial inspection. EfficientAD trades the memory bank for a small CNN trained via student-teacher distillation, hitting 99.0+ image-AUROC on MVTec-AD while running at sub-millisecond inference on a T4.

| | PatchCore | EfficientAD-S | EfficientAD-M |
|---|---|---|---|
| Image-AUROC (mean) | 99.1 | 98.8 | 99.1 |
| Inference latency | ~50ms | <1ms | ~2ms |
| Backbone params | 68M (WRN-50) | 1.4M | 8.4M |

## Goal

Match the paper's reported numbers within ±0.5 points on MVTec-AD across all 15 categories:

| | Image-AUROC | Pixel-AUROC | Pixel-AUPRO |
|---|---|---|---|
| Paper EfficientAD-S (mean) | 98.8 | 96.8 | 92.6 |
| Paper EfficientAD-M (mean) | 99.1 | 97.6 | 93.6 |
| This repo (S) | TBD | TBD | TBD |
| This repo (M) | TBD | TBD | TBD |

## Roadmap

- [ ] Repo scaffold, configs, dataset loader (reused from patchcore-reproduced)
- [ ] Patch description network (PDN) — small student/teacher backbone
- [ ] ImageNet-pretrained teacher distillation
- [ ] Autoencoder for global anomaly detection
- [ ] Combined local + global anomaly map
- [ ] Single-category and all-category runners
- [ ] Empirical reproduction across 15 categories
- [ ] Walkthrough notebook with qualitative figures
- [ ] Medium walkthrough post

## References

- Batzner et al., *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies*, WACV 2024
- Bergmann et al., *MVTec-AD*, CVPR 2019
- Official EfficientAD repo: https://github.com/nelson1425/EfficientAD

## License

MIT
