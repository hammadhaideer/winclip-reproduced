import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from anomalib.models.image.winclip import WinClipModel
from src.datasets import MVTecAD, MVTEC_CATEGORIES, VisA, VISA_CATEGORIES
from src.metrics import image_auroc, pixel_auroc, per_region_overlap


def get_dataset_class(name):
    if name == "mvtec_ad":
        return MVTecAD, MVTEC_CATEGORIES
    if name == "visa":
        return VisA, VISA_CATEGORIES
    raise ValueError(name)


def get_dataset_root(dataset_name):
    import os
    root = os.environ.get("WINCLIP_DATA_ROOT")
    if root is None:
        raise EnvironmentError("Set WINCLIP_DATA_ROOT")
    return Path(root) / dataset_name


@torch.no_grad()
def run_category(dataset_name, category, shot, output_dir, device):
    DatasetCls, _ = get_dataset_class(dataset_name)
    data_root = get_dataset_root(dataset_name)

    test_ds = DatasetCls(str(data_root), category, split="test", input_size=240, resize=240)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    reference_images = None
    if shot > 0:
        train_ds = DatasetCls(str(data_root), category, split="train", input_size=240, resize=240)
        reference_images = torch.stack([train_ds[i]["image"] for i in range(min(shot, len(train_ds)))])

    model = WinClipModel(
        class_name=category.replace("_", " "),
        reference_images=reference_images,
        scales=(2, 3),
        apply_transform=False,
    ).to(device).eval()

    scores, labels, maps, masks = [], [], [], []
    t0 = time.time()

    for batch in tqdm(test_loader, desc=f"{dataset_name}/{category} {shot}-shot"):
        images = batch["image"].to(device)
        out = model(images)
        scores.extend(out.pred_score.detach().cpu().numpy().tolist())
        labels.extend(batch["label"].numpy().tolist())
        maps.extend(out.anomaly_map.detach().cpu().numpy())
        masks.extend(batch["mask"].numpy())

    img = image_auroc(np.array(scores), np.array(labels))
    pix = pixel_auroc(maps, masks)
    pro = per_region_overlap(maps, masks)

    result = {
        "dataset": dataset_name,
        "category": category,
        "shot": shot,
        "image_auroc": round(img * 100, 2),
        "pixel_auroc": round(pix * 100, 2),
        "pixel_aupro": round(pro * 100, 2),
        "n_test": len(test_ds),
        "runner": "anomalib.WinClipModel",
        "timing_seconds": round(time.time() - t0, 1),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{dataset_name}_{category}_{shot}shot.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"image_auroc={result['image_auroc']:.2f} pixel_auroc={result['pixel_auroc']:.2f} pixel_aupro={result['pixel_aupro']:.2f}")
    print(f"saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mvtec_ad", "visa"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--shot", type=int, default=0)
    parser.add_argument("--output_dir", default="results_anomalib")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_category(args.dataset, args.category, args.shot, args.output_dir, args.device)


if __name__ == "__main__":
    main()
