# winclip-reproduced

> Clean reproduction of **WinCLIP** (Jeong et al., CVPR 2023) on MVTec-AD and VisA — zero-shot and few-normal-shot anomaly classification and segmentation with CLIP.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Paper: [WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation](https://arxiv.org/abs/2303.14814) - Jeong et al., CVPR 2023

## What this is

End-to-end reproduction of WinCLIP, the first strong CLIP-based zero-shot anomaly detector on MVTec-AD and VisA. Implements the compositional prompt ensemble, multi-scale window aggregation, and the few-normal-shot extension WinCLIP+.

This is the second reproduction in a series of visual anomaly detection methods I'm building toward UniVAD (CVPR 2025), my main thesis baseline. WinCLIP is one of UniVAD's six comparison methods in Table 1.

## Status

In active development. Code lands over the coming week, results follow once GPU access returns.

## Why WinCLIP

WinCLIP is one of two foundation-model-based methods on UniVAD's comparison list (alongside AnomalyGPT). It shifted visual anomaly detection from "train a custom model per object" to "use a pretrained vision-language model with the right prompts" making it the conceptual bridge between PatchCore-era frozen-ImageNet-backbone methods and the current foundation-model era of AD.

| | Zero-shot Image-AUROC (MVTec-AD) | 1-shot Image-AUROC (MVTec-AD) |
|---|---|---|
| WinCLIP | 91.8 | — |
| WinCLIP+ | — | 93.1 |

(Numbers from the WinCLIP paper.)

## Goal

Match the paper's reported numbers within ±0.5 points:

| Setting | Dataset | Image-AUROC | Pixel-AUROC |
|---|---|---|---|
| Zero-shot | MVTec-AD | 91.8 → TBD | 85.1 → TBD |
| Zero-shot | VisA | 78.1 → TBD | 79.6 → TBD |
| 1-normal-shot | MVTec-AD | 93.1 → TBD | 95.2 → TBD |
| 1-normal-shot | VisA | 83.8 → TBD | 96.4 → TBD |

## Installation

Clone and create the environment:

```bash
git clone https://github.com/hammadhaideer/winclip-reproduced.git
cd winclip-reproduced
conda env create -f environment.yml
conda activate winclip
```

Or with pip in an existing environment:

```bash
pip install -r requirements.txt
```

## Datasets

This repo evaluates on **MVTec-AD** and **VisA**. Download both, then point the code to a parent directory containing them:

```bash
export WINCLIP_DATA_ROOT=/path/to/anomaly_datasets
```

Expected layout:

```text
$WINCLIP_DATA_ROOT/
├── mvtec_ad/
│   ├── bottle/
│   │   ├── train/good/*.png
│   │   ├── test/<defect>/*.png
│   │   └── ground_truth/<defect>/*_mask.png
│   ├── cable/
│   └── ... (15 categories)
└── visa/
    ├── candle/
    ├── ... (12 categories)
    └── split_csv/1cls.csv
```

| Dataset | Source | Categories |
|---|---|---|
| MVTec-AD | https://www.mvtec.com/company/research/datasets/mvtec-ad | 15 |
| VisA | https://github.com/amazon-science/spot-diff | 12 |

VisA is distributed with a split CSV file (`split_csv/1cls.csv`) that defines the train/test partition. The loader reads from this CSV — make sure it's present at the path above.

## Run

Single category, zero-shot:

```bash
python scripts/run_winclip.py --config configs/default.yaml \
    --dataset mvtec_ad --category bottle --shot 0
```

Single category, 1-normal-shot:

```bash
python scripts/run_winclip.py --config configs/default.yaml \
    --dataset mvtec_ad --category bottle --shot 1
```

Sweep all categories on both datasets, both 0-shot and 1-shot (~1 hour on a single GPU):

```bash
python scripts/run_all.py --datasets mvtec_ad visa --shots 0 1
```

Full paper reproduction (108 runs, ~1 hour total):

```bash
python scripts/run_all.py --datasets mvtec_ad visa --shots 0 1 2 4
```

Build the comparison table after runs complete:

```bash
python scripts/aggregate_results.py --results_dir results --shots 0 1
```

Per-experiment results land in `results/<dataset>_<category>_<shot>shot.json`. The aggregated table prints to stdout, comparing your numbers against the paper's reported values per (dataset, shot) combination.

## Roadmap

- [ ] Repo scaffold, configs, dataset loader (reused from patchcore-reproduced)
- [ ] CLIP ViT-B/16+ backbone integration via `open_clip`
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
