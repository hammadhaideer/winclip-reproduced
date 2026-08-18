# WinCLIP pixel-localization diagnostic

This repository reproduces WinCLIP zero-shot image-level AUROC closely on MVTec-AD and VisA, but pixel-level localization remains below the paper-reported numbers.

## Observed results

Using Anomalib 2.6.0, OpenCLIP 3.3.0, and the ViT-B-16-plus-240 LAION-400M checkpoint:

| Dataset | Paper image AUROC | Reproduced image AUROC | Paper pixel AUROC | Reproduced pixel AUROC |
|---|---:|---:|---:|---:|
| MVTec-AD | 91.8 | 89.96 | 85.1 | 67.02 |
| VisA | 78.1 | 75.36 | 79.6 | 66.46 |

All values are percentages and macro-averaged across categories.

## Diagnostic finding

The unmodified Anomalib `WinClipModel` produced nearly constant zero-shot anomaly maps with the current OpenCLIP stack. Mean-pooling selected window patch tokens restored spatial variation, but map polarity was category-dependent:

- MVTec hazelnut: inverted map improved pixel AUROC to 84.94.
- MVTec transistor: inverted map degraded pixel AUROC to 14.97, while the non-inverted map was 86.64.

Therefore, a single global polarity inversion is not a valid final fix.

## Current conclusion

The repository should report image-level reproduction as near-complete and pixel-level localization as under investigation. Pixel results are included for transparency but should not be presented as a full paper-level reproduction.
