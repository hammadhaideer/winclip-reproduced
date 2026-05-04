import numpy as np
from skimage.measure import label, regionprops


def per_region_overlap(score_maps, gt_masks, max_fpr=0.3, num_thresholds=200):
    score_maps = np.stack(score_maps).astype(np.float32)
    gt_masks = np.stack(gt_masks).astype(np.uint8)

    flat_scores_neg = score_maps[gt_masks == 0]
    if flat_scores_neg.size == 0:
        return 0.0

    thresholds = np.quantile(
        flat_scores_neg,
        np.linspace(1.0 - max_fpr * 1.5, 1.0, num_thresholds),
    )

    region_components = []
    for i in range(gt_masks.shape[0]):
        lbl = label(gt_masks[i])
        for region in regionprops(lbl):
            region_components.append((i, region.coords))

    fprs = []
    pros = []
    for t in thresholds[::-1]:
        pred = (score_maps >= t)
        fpr = pred[gt_masks == 0].mean()
        if fpr > max_fpr:
            continue

        if not region_components:
            continue
        overlaps = [
            pred[i][coords[:, 0], coords[:, 1]].mean()
            for i, coords in region_components
        ]
        pros.append(float(np.mean(overlaps)))
        fprs.append(float(fpr))

    if not fprs:
        return 0.0

    fprs = np.array(fprs)
    pros = np.array(pros)
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    aupro = np.trapz(pros, fprs) / max_fpr
    return float(aupro)
