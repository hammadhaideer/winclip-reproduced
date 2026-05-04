import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import MVTEC_CATEGORIES, VISA_CATEGORIES


PAPER_NUMBERS = {
    ("mvtec_ad", 0): {"image_auroc": 91.8, "pixel_auroc": 85.1},
    ("mvtec_ad", 1): {"image_auroc": 93.1, "pixel_auroc": 95.2},
    ("mvtec_ad", 2): {"image_auroc": 94.4, "pixel_auroc": 96.0},
    ("mvtec_ad", 4): {"image_auroc": 95.2, "pixel_auroc": 96.2},
    ("visa", 0): {"image_auroc": 78.1, "pixel_auroc": 79.6},
    ("visa", 1): {"image_auroc": 83.8, "pixel_auroc": 96.4},
    ("visa", 2): {"image_auroc": 84.6, "pixel_auroc": 96.8},
    ("visa", 4): {"image_auroc": 87.3, "pixel_auroc": 97.2},
}


def load_results(results_dir, dataset, shot):
    categories = MVTEC_CATEGORIES if dataset == "mvtec_ad" else VISA_CATEGORIES
    rows = []
    for cat in categories:
        path = Path(results_dir) / f"{dataset}_{cat}_{shot}shot.json"
        if not path.exists():
            rows.append((cat, None, None, None))
            continue
        with open(path) as f:
            r = json.load(f)
        rows.append((
            cat,
            r["image_auroc"],
            r["pixel_auroc"],
            r["pixel_aupro"],
        ))
    return rows


def print_table(dataset, shot, rows):
    paper = PAPER_NUMBERS.get((dataset, shot))
    paper_img = paper["image_auroc"] if paper else None
    paper_pix = paper["pixel_auroc"] if paper else None

    print(f"\n=== {dataset} | {shot}-shot ===")
    print(f"{'Category':<14} {'Img-AUROC':>10} {'Pix-AUROC':>10} {'Pix-AUPRO':>10}")
    print("-" * 48)

    img_vals, pix_vals, aupro_vals = [], [], []
    for cat, im, px, ap in rows:
        if im is None:
            print(f"{cat:<14} {'—':>10} {'—':>10} {'—':>10}")
        else:
            print(f"{cat:<14} {im:>10.2f} {px:>10.2f} {ap:>10.2f}")
            img_vals.append(im)
            pix_vals.append(px)
            aupro_vals.append(ap)

    if img_vals:
        print("-" * 48)
        print(f"{'mean':<14} {sum(img_vals)/len(img_vals):>10.2f} "
              f"{sum(pix_vals)/len(pix_vals):>10.2f} "
              f"{sum(aupro_vals)/len(aupro_vals):>10.2f}")
        if paper:
            print(f"{'paper':<14} {paper_img:>10.2f} {paper_pix:>10.2f} {'—':>10}")
            delta_img = sum(img_vals)/len(img_vals) - paper_img
            delta_pix = sum(pix_vals)/len(pix_vals) - paper_pix
            print(f"{'delta':<14} {delta_img:>+10.2f} {delta_pix:>+10.2f} {'':>10}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--datasets", nargs="+", default=["mvtec_ad", "visa"])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()

    for dataset in args.datasets:
        for shot in args.shots:
            rows = load_results(args.results_dir, dataset, shot)
            print_table(dataset, shot, rows)


if __name__ == "__main__":
    main()
