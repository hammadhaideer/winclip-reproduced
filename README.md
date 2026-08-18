# WinCLIP Reproduced

Independent reproduction workspace for **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation** (CVPR 2023).

This repository is part of a focused visual anomaly-detection reproduction series around CLIP-based industrial anomaly detection. The goal is to reproduce the major methods compared with AF-CLIP, document what matches, and keep transparent debugging records when a metric does not yet match the paper.

## Current status

The current Anomalib-backed zero-shot run gives a near image-level reproduction, but pixel-level localization is not yet paper-correct in this environment. A diagnostic patch restored spatial variation in the zero-shot maps, but the map polarity is category-dependent, so a single global inversion is not a valid final fix.

| Dataset | Shot | Image AUROC | Paper Image AUROC | Delta | Pixel AUROC | Paper Pixel AUROC | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| MVTec-AD | 0 | 89.96 | 91.80 | -1.84 | 67.02 | 85.10 | Image near; pixel unresolved |
| VisA | 0 | 75.36 | 78.10 | -2.74 | 66.46 | 79.60 | Image near; pixel unresolved |

All values are percentages and macro-averaged across categories.

## What is included

- Dataset loaders for MVTec-AD and VisA.
- A local prototype WinCLIP implementation.
- An Anomalib-backed WinCLIP runner.
- Zero-shot category-level JSON results for MVTec-AD and VisA.
- Aggregation script for paper comparison.
- Documentation of current reproduction status and known limitations.

## What is not claimed yet

This repository does **not** yet claim full WinCLIP reproduction because the pixel-level localization numbers are below the CVPR 2023 paper. The current evidence supports:

- near reproduction of zero-shot image-level AUROC;
- successful infrastructure for running all MVTec-AD and VisA categories;
- an identified pixel-localization issue requiring further debugging.

See `docs/WINCLIP_PIXEL_DIAGNOSTIC.md` for the pixel-map diagnostic.

## Reproduction results

Aggregate table:

```text
results/summary.csv
```

Category-level Anomalib zero-shot results:

```text
results/raw/per_category_anomalib_zeroshot.csv
```

Diagnostic run folders are ignored by Git by default.

## Setup

```bash
conda env create -f environment.yml
conda activate winclip
pip install anomalib open_clip_torch
```

The server used for the current reproduction had no direct access to HuggingFace, so the OpenCLIP checkpoint was downloaded separately and provided locally.

Required checkpoint:

```text
timm/vit_base_patch16_plus_clip_240.laion400m_e31/open_clip_pytorch_model.bin
```

Local SHA256 used:

```text
fa8eec9aff58e9215b9b44a977038179712694d3fc3a73eba62546bcff13deb3
```

The checkpoint is **not** redistributed in this repository.

## Datasets

Set:

```bash
export WINCLIP_DATA_ROOT=/path/to/winclip_data
```

Expected layout:

```text
WINCLIP_DATA_ROOT/
├── mvtec_ad/
└── visa/
```

## Run

Single Anomalib-backed category:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_winclip_anomalib.py \
  --dataset mvtec_ad \
  --category bottle \
  --shot 0 \
  --output_dir results_anomalib_zs
```

Full zero-shot sweep:

```bash
python scripts/run_all_anomalib.py \
  --datasets mvtec_ad visa \
  --shots 0 \
  --output_dir results_anomalib_zs
```

Aggregate:

```bash
python scripts/aggregate_results.py --results_dir results_anomalib_zs --shots 0
```

## Known issue

With Anomalib 2.6.0 and OpenCLIP 3.3.0 in this environment, zero-shot image scores are meaningful, but pixel localization remains below the paper target. The original Anomalib window path produced nearly constant maps; replacing CLS pooling with selected window patch-token pooling restored spatial variation, but polarity varied by category.

This repository keeps that finding visible instead of hiding it, because the purpose of the project is transparent reproduction.

## Reproduction series

Focused AF-CLIP comparison-chain reproduction:

- [x] [AF-CLIP reproduced](https://github.com/hammadhaideer/af-clip-reproduced)
- [x] [AnomalyCLIP reproduced](https://github.com/hammadhaideer/anomalyclip-reproduced)
- [~] WinCLIP reproduced — image-level near reproduction; pixel localization under debugging
- [ ] VAND / APRIL-GAN
- [ ] AdaCLIP
- [ ] AA-CLIP

## References

- Jeong et al., **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation**, CVPR 2023.
- Radford et al., **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021.
- Bergmann et al., **The MVTec Anomaly Detection Dataset**, CVPR 2019.
- Zou et al., **Spot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation**, ECCV 2022.

## License

The independently written wrapper code and documentation in this repository are released under the MIT License. This license does not apply to WinCLIP, Anomalib, OpenCLIP, CLIP checkpoints, or the benchmark datasets.
