import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import MVTecAD, MVTEC_CATEGORIES, VisA, VISA_CATEGORIES, get_data_root
from src.methods import WinCLIP
from src.metrics import image_auroc, pixel_auroc, per_region_overlap


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_dataset_class(name):
    if name == "mvtec_ad":
        return MVTecAD, MVTEC_CATEGORIES
    elif name == "visa":
        return VisA, VISA_CATEGORIES
    else:
        raise ValueError(f"Unknown dataset: {name}")


def get_dataset_root(dataset_name):
    import os
    root = os.environ.get("WINCLIP_DATA_ROOT")
    if root is None:
        raise EnvironmentError(
            "Set WINCLIP_DATA_ROOT to your dataset root containing mvtec_ad/ and visa/."
        )
    if dataset_name == "mvtec_ad":
        return Path(root) / "mvtec_ad"
    elif dataset_name == "visa":
        return Path(root) / "visa"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collect_reference_images(train_loader, n_shot, device):
    images = []
    for batch in train_loader:
        images.append(batch["image"])
        if sum(x.shape[0] for x in images) >= n_shot:
            break
    return torch.cat(images, dim=0)[:n_shot].to(device)


def run_category(dataset_name, category, cfg):
    device = cfg["eval"]["device"]
    seed = cfg["eval"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    DatasetCls, _ = get_dataset_class(dataset_name)
    data_root = get_dataset_root(dataset_name)
    n_shot = cfg["winclip"]["shot"]

    test_ds = DatasetCls(
        str(data_root), category, split="test",
        input_size=cfg["dataset"]["input_size"],
        resize=cfg["dataset"]["resize"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["eval"]["batch_size"],
        shuffle=False, num_workers=cfg["eval"]["num_workers"], pin_memory=True,
    )

    winclip = WinCLIP(
        model_name=cfg["backbone"]["name"],
        pretrained=cfg["backbone"]["pretrained"],
        scales=tuple(cfg["winclip"]["scales"]),
        device=device,
    )

    t0 = time.time()
    winclip.setup_text(category.replace("_", " "))
    t_text = time.time() - t0

    t_ref = 0.0
    if n_shot > 0:
        train_ds = DatasetCls(
            str(data_root), category, split="train",
            input_size=cfg["dataset"]["input_size"],
            resize=cfg["dataset"]["resize"],
        )
        train_loader = DataLoader(
            train_ds, batch_size=n_shot, shuffle=False, num_workers=2,
        )
        t0 = time.time()
        ref_images = collect_reference_images(train_loader, n_shot, device)
        winclip.setup_reference(ref_images)
        t_ref = time.time() - t0

    image_scores = []
    labels = []
    score_maps = []
    gt_masks = []

    t0 = time.time()
    for batch in tqdm(test_loader, desc=f"{dataset_name}/{category} {n_shot}-shot"):
        img_score, score_map = winclip.score_image(
            batch["image"], target_size=cfg["dataset"]["input_size"],
        )
        image_scores.extend(img_score.cpu().numpy().tolist())
        labels.extend(batch["label"].cpu().numpy().tolist())
        score_maps.extend(score_map.cpu().numpy())
        gt_masks.extend(batch["mask"].cpu().numpy())
    t_score = time.time() - t0

    img_auroc = image_auroc(np.array(image_scores), np.array(labels))
    pix_auroc = pixel_auroc(score_maps, gt_masks)
    aupro = per_region_overlap(score_maps, gt_masks)

    results = {
        "dataset": dataset_name,
        "category": category,
        "shot": n_shot,
        "image_auroc": round(img_auroc * 100, 2),
        "pixel_auroc": round(pix_auroc * 100, 2),
        "pixel_aupro": round(aupro * 100, 2),
        "n_test": len(test_ds),
        "timing_seconds": {
            "text_setup": round(t_text, 1),
            "reference_setup": round(t_ref, 1),
            "test_scoring": round(t_score, 1),
        },
        "config": cfg,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True, choices=["mvtec_ad", "visa"])
    parser.add_argument("--category", required=True)
    parser.add_argument("--shot", type=int, default=None,
                        help="Override config: 0=zero-shot, 1/2/4=few-normal-shot")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.shot is not None:
        cfg["winclip"]["shot"] = args.shot
    cfg["dataset"]["name"] = args.dataset

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n=== {args.dataset}/{args.category} | {cfg['winclip']['shot']}-shot ===")
    res = run_category(args.dataset, args.category, cfg)

    out_name = f"{args.dataset}_{args.category}_{cfg['winclip']['shot']}shot.json"
    out_path = Path(args.output_dir) / out_name
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"  image_auroc={res['image_auroc']:.2f}  "
          f"pixel_auroc={res['pixel_auroc']:.2f}  "
          f"pixel_aupro={res['pixel_aupro']:.2f}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
