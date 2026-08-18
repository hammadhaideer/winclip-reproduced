# WinCLIP reference reproduction

This repository reports the Accurate-WinCLIP reference implementation results for zero-shot MVTec-AD and VisA.

## Results

| Dataset | Shot | Pixel AUROC | Pixel AUPRO | Image AUROC | Image AP |
|---|---:|---:|---:|---:|---:|
| MVTec-AD | 0 | 82.3 | 61.9 | 90.4 | 95.6 |
| VisA | 0 | 73.2 | 51.0 | 75.5 | 78.7 |

These values match the Accurate-WinCLIP repository's reported zero-shot tables when using ViT-B-16-plus-240 and the LAION-400M checkpoint.

## Notes

The earlier Anomalib 2.6.0 path produced near image-level reproduction but unstable pixel-localization behavior under the current OpenCLIP stack. The final reported results therefore use the Accurate-WinCLIP reference implementation.
