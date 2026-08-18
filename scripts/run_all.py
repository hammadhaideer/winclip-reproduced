import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import MVTEC_CATEGORIES, VISA_CATEGORIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=["mvtec_ad", "visa"],
                        choices=["mvtec_ad", "visa"])
    parser.add_argument("--shots", nargs="+", type=int, default=[0, 1],
                        help="Shot settings to run (e.g., --shots 0 1 2 4)")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    dataset_categories = {
        "mvtec_ad": MVTEC_CATEGORIES,
        "visa": VISA_CATEGORIES,
    }

    total = sum(len(dataset_categories[d]) for d in args.datasets) * len(args.shots)
    done = 0

    for dataset in args.datasets:
        for category in dataset_categories[dataset]:
            for shot in args.shots:
                done += 1
                cmd = [
                    sys.executable, "scripts/run_winclip.py",
                    "--config", args.config,
                    "--dataset", dataset,
                    "--category", category,
                    "--shot", str(shot),
                    "--output_dir", args.output_dir,
                ]
                print(f"\n>>> [{done}/{total}] {' '.join(cmd)}")
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
