import os
import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4",
    "pipe_fryum",
]


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VisA(Dataset):
    def __init__(self, root, category, split="train", input_size=240, resize=256,
                 mean=CLIP_MEAN, std=CLIP_STD, csv_filename="split_csv/1cls.csv"):
        self.root = Path(root)
        self.category = category
        if not (self.root / category).exists():
            raise FileNotFoundError(f"Category not found: {self.root / category}")

        self.split = split
        self.input_size = input_size
        self.csv_path = self.root / csv_filename

        self.image_transform = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.mask_transform = T.Compose([
            T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(input_size),
            T.PILToTensor(),
        ])

        self.samples = self._load_samples()

    def _load_samples(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"VisA split CSV not found: {self.csv_path}. "
                f"Download VisA from https://github.com/amazon-science/spot-diff and "
                f"ensure split_csv/1cls.csv is in the dataset root."
            )

        samples = []
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["object"] != self.category:
                    continue
                if row["split"] != self.split:
                    continue

                img_rel = row["image"]
                mask_rel = row["mask"]
                label = 0 if row["label"] == "normal" else 1

                img_path = self.root / img_rel
                mask_path = self.root / mask_rel if mask_rel else None

                samples.append({
                    "image": img_path,
                    "label": label,
                    "mask": mask_path,
                    "defect": "good" if label == 0 else "anomaly",
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        img = self.image_transform(img)

        if s["mask"] is not None and s["mask"].exists():
            mask = Image.open(s["mask"]).convert("L")
            mask = self.mask_transform(mask)
            mask = (mask > 0).float().squeeze(0)
        else:
            mask = torch.zeros(self.input_size, self.input_size, dtype=torch.float32)

        return {
            "image": img,
            "label": s["label"],
            "mask": mask,
            "defect": s["defect"],
            "category": self.category,
            "path": str(s["image"]),
        }
