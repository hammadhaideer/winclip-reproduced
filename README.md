# WinCLIP Reproduced

Independent reproduction workspace for **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation** (CVPR 2023).

This repository is part of a focused visual anomaly-detection reproduction series around CLIP-based industrial anomaly detection. The goal is to reproduce the major methods compared with AF-CLIP, document what matches, and keep transparent debugging records when a metric does not match the original paper exactly.

## Current status

The final reported zero-shot results use the Accurate-WinCLIP reference implementation with ViT-B-16-plus-240 and the LAION-400M checkpoint. The reproduced numbers match the Accurate-WinCLIP reference tables.

| Dataset | Shot | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP | Status |
|---|---:|---:|---:|---:|---:|---|
| MVTec-AD | 0 | 82.3 | 61.9 | 90.4 | 95.6 | Reference reproduced |
| VisA | 0 | 73.2 | 51.0 | 75.5 | 78.7 | Reference reproduced |

All values are percentages and macro-averaged across categories.

## What is included

- Dataset loaders for MVTec-AD and VisA.
- A local prototype WinCLIP implementation.
- An Anomalib-backed diagnostic runner.
- Accurate-WinCLIP reference zero-shot logs for MVTec-AD and VisA.
- Aggregation script for paper/reference comparison.
- Documentation of the Anomalib pixel-localization diagnostic.

## Reproduction results

Aggregate table:

```text
results/summary.csv
```

Raw reference logs:

```text
results/raw/mvtec_accurate_winclip_zs.log
results/raw/visa_accurate_winclip_zs.log
```

Reference note:

```text
docs/REFERENCE_REPRODUCTION.md
```

The Anomalib diagnostic note is retained at:

```text
docs/WINCLIP_PIXEL_DIAGNOSTIC.md
```

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

Single Anomalib-backed diagnostic category:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_winclip_anomalib.py \
  --dataset mvtec_ad \
  --category bottle \
  --shot 0 \
  --output_dir results_anomalib_zs
```

Full Anomalib diagnostic sweep:

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

## Notes on implementation paths

The earlier Anomalib 2.6.0 path produced near image-level reproduction but unstable pixel-localization behavior under the current OpenCLIP stack. The final reported numbers therefore use the Accurate-WinCLIP reference implementation, while the Anomalib diagnostic files remain in the repository for transparency.

## Reproduction series

Focused AF-CLIP comparison-chain reproduction:

- [x] [AF-CLIP reproduced](https://github.com/hammadhaideer/af-clip-reproduced)
- [x] [AnomalyCLIP reproduced](https://github.com/hammadhaideer/anomalyclip-reproduced)
- [x] WinCLIP reproduced — Accurate-WinCLIP reference implementation results
- [ ] VAND / APRIL-GAN
- [ ] AdaCLIP
- [ ] AA-CLIP

## References

- Jeong et al., **WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation**, CVPR 2023.
- Accurate-WinCLIP PyTorch reference implementation.
- Radford et al., **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021.
- Bergmann et al., **The MVTec Anomaly Detection Dataset**, CVPR 2019.
- Zou et al., **Spot-the-Difference Self-supervised Pre-training for Anomaly Detection and Segmentation**, ECCV 2022.

## License

The independently written wrapper code and documentation in this repository are released under the MIT License. This license does not apply to WinCLIP, Accurate-WinCLIP, Anomalib, OpenCLIP, CLIP checkpoints, or the benchmark datasets.
