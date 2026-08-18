import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper",
]


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class MVTecAD(Dataset):
    def __init__(self, root, category, split="train", input_size=240, resize=256,
                 mean=CLIP_MEAN, std=CLIP_STD):
        self.root = Path(root) / category
        if not self.root.exists():
            raise FileNotFoundError(f"Category not found: {self.root}")

        self.split = split
        self.input_size = input_size

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
        samples = []
        if self.split == "train":
            train_dir = self.root / "train" / "good"
            for p in sorted(train_dir.glob("*.png")):
                samples.append({
                    "image": p, "label": 0, "mask": None, "defect": "good",
                })
        else:
            test_dir = self.root / "test"
            mask_dir = self.root / "ground_truth"
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect = defect_dir.name
                label = 0 if defect == "good" else 1
                for img_path in sorted(defect_dir.glob("*.png")):
                    mask_path = None
                    if defect != "good":
                        candidate = mask_dir / defect / f"{img_path.stem}_mask.png"
                        if candidate.exists():
                            mask_path = candidate
                    samples.append({
                        "image": img_path,
                        "label": label,
                        "mask": mask_path,
                        "defect": defect,
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image"]).convert("RGB")
        img = self.image_transform(img)

        if s["mask"] is not None:
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
            "category": s["image"].parent.parent.parent.name,
            "path": str(s["image"]),
        }


def get_data_root():
    root = os.environ.get("WINCLIP_DATA_ROOT")
    if root is None:
        raise EnvironmentError(
            "Set WINCLIP_DATA_ROOT to your dataset root directory containing mvtec_ad/ and visa/. "
            "Example: export WINCLIP_DATA_ROOT=/data/anomaly"
        )
    return os.path.join(root, "mvtec_ad")
