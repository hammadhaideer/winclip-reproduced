import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.datasets import MVTEC_CATEGORIES, VISA_CATEGORIES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["mvtec_ad", "visa"], choices=["mvtec_ad", "visa"])
    parser.add_argument("--shots", nargs="+", type=int, default=[0])
    parser.add_argument("--output_dir", default="results_anomalib")
    args = parser.parse_args()

    cats = {"mvtec_ad": MVTEC_CATEGORIES, "visa": VISA_CATEGORIES}
    total = sum(len(cats[d]) for d in args.datasets) * len(args.shots)
    done = 0
    for dataset in args.datasets:
        for category in cats[dataset]:
            for shot in args.shots:
                done += 1
                cmd = [
                    sys.executable, "scripts/run_winclip_anomalib.py",
                    "--dataset", dataset,
                    "--category", category,
                    "--shot", str(shot),
                    "--output_dir", args.output_dir,
                ]
                print(f"\n>>> [{done}/{total}] {' '.join(cmd)}", flush=True)
                subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

if __name__ == "__main__":
    main()
