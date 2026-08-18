import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(scores, labels):
    return float(roc_auc_score(labels, scores))


def pixel_auroc(score_maps, gt_masks):
    flat_scores = np.concatenate([s.flatten() for s in score_maps])
    flat_gt = np.concatenate([g.flatten() for g in gt_masks]).astype(int)
    return float(roc_auc_score(flat_gt, flat_scores))
