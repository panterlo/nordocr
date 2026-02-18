"""
Dataset loader for Nordic OCR training data.

Reads the TSV + images format produced by the C# data preparation pipeline:
    images/000001.png
    labels.tsv: filename\ttext\tconfidence\tsource_fid

Supports mixed datasets (auto-labeled + spot-check) with per-source weighting.
"""

import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T
from PIL import Image

from charset import NordicTokenizer


class AspectPreservingResize:
    """Resize to target height preserving aspect ratio, then pad width to max_width.

    For images already at height=32 (from C# export), this is a no-op resize.
    Images wider than max_width are scaled down to fit (height shrinks proportionally,
    then padded back to target height). Images narrower are right-padded.

    Padding uses gray (pixel value 128) which maps to 0.0 after normalization
    with mean=0.5, std=0.5 — matching the crop_resize_norm.cu kernel behavior.
    """

    def __init__(self, target_h, max_w, interpolation=T.InterpolationMode.BICUBIC):
        self.target_h = target_h
        self.max_w = max_w
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size  # PIL is (width, height)

        # Scale so height = target_h, preserving aspect ratio
        scale = self.target_h / h
        new_w = round(w * scale)

        if new_w > self.max_w:
            # Image too wide — stretch to (target_h, max_w).
            # This matches inference behavior: crop_resize_norm.cu maps the
            # full source region into dst_w pixels when dst_w is clamped.
            img = img.resize((self.max_w, self.target_h), Image.BICUBIC)
            return img

        # Resize preserving aspect ratio, then right-pad to max_w with gray.
        # Gray (128) maps to ~0.0 after Normalize(0.5, 0.5), matching the
        # 0.0f padding written by crop_resize_norm.cu for pixels beyond dst_w.
        img = img.resize((new_w, self.target_h), Image.BICUBIC)
        padded = Image.new("RGB", (self.max_w, self.target_h), (128, 128, 128))
        padded.paste(img, (0, 0))
        return padded

    def __repr__(self):
        return f"AspectPreservingResize(h={self.target_h}, max_w={self.max_w})"


class NordicOCRDataset(Dataset):
    """Dataset for cropped text line images with TSV labels."""

    def __init__(
        self,
        data_dir,
        tokenizer=None,
        img_size=(32, 384),
        max_label_len=128,
        augment=False,
    ):
        """
        Args:
            data_dir: Path to directory containing images/ and labels.tsv
            tokenizer: NordicTokenizer instance (created if None)
            img_size: (height, max_width) — images are aspect-ratio resized + padded
            max_label_len: Maximum text length (chars, not tokens)
            augment: Whether to apply training augmentation
        """
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.tokenizer = tokenizer or NordicTokenizer()
        self.img_size = img_size
        self.max_label_len = max_label_len

        # Load labels
        self.samples = self._load_labels()

        # Image transforms — aspect-preserving resize + pad first,
        # so augmentations have stable geometry
        transform_list = [
            AspectPreservingResize(img_size[0], img_size[1]),
        ]
        if augment:
            transform_list.extend([
                T.RandomRotation(3),
                T.RandomPerspective(distortion_scale=0.1, p=0.3),
                T.ColorJitter(brightness=0.3, contrast=0.3),
            ])
        transform_list.extend([
            T.ToTensor(),
            # IMPORTANT: Must be T.Normalize(0.5, 0.5), mapping [0,1] to [-1,1].
            # The inference CUDA kernel (crop_resize_norm.cu) uses matching constants:
            #   NORM_MEAN = {0.5, 0.5, 0.5}, NORM_STD = {0.5, 0.5, 0.5}
            # Do NOT change to ImageNet stats — it will break inference silently.
            #
            # Padding pixels (128/255 ≈ 0.502) normalize to ~0.0, matching the
            # 0.0f padding in crop_resize_norm.cu for inference.
            T.Normalize(0.5, 0.5),
        ])
        self.transform = T.Compose(transform_list)

    def _load_labels(self):
        """Load labels.tsv file using plain tab-split (csv.DictReader chokes on embedded quotes)."""
        samples = []
        tsv_path = self.data_dir / "labels.tsv"
        charset_set = set(self.tokenizer.charset)

        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < len(header):
                    continue
                row = dict(zip(header, fields))
                filename = row["filename"]
                text = row.get("text") or row.get("verified_text") or ""
                # Filter to charset-valid characters and enforce max length
                filtered = "".join(c for c in text if c in charset_set)
                if filtered and len(filtered) <= self.max_label_len:
                    samples.append({
                        "filename": filename,
                        "text": filtered,
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image
        img_path = self.img_dir / sample["filename"]
        img = Image.open(img_path).convert("RGB")

        # Transform: resize to exact (H, W) + normalize to [-1, 1]
        img = self.transform(img)

        return img, sample["text"]


def collate_fn(batch):
    """Collate function for DataLoader — returns stacked images + list of label strings."""
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(labels)


def build_train_dataset(
    auto_labeled_dir,
    spot_check_dir=None,
    tokenizer=None,
    img_size=(32, 384),
    max_label_len=128,
    spot_check_oversample=4,
):
    """
    Build combined training dataset with oversampling.

    Args:
        auto_labeled_dir: Path to Tesseract auto-labeled data
        spot_check_dir: Path to verified spot-check cutouts (optional)
        tokenizer: NordicTokenizer instance
        img_size: (height, width) for image resize
        max_label_len: Maximum text length (chars)
        spot_check_oversample: How many times to repeat spot-check data

    Returns:
        ConcatDataset
    """
    tokenizer = tokenizer or NordicTokenizer()

    datasets = []

    # Auto-labeled dataset
    auto_ds = NordicOCRDataset(
        auto_labeled_dir,
        tokenizer=tokenizer,
        img_size=img_size,
        max_label_len=max_label_len,
        augment=True,
    )
    datasets.append(auto_ds)
    print(f"Auto-labeled: {len(auto_ds)} samples")

    # Spot-check dataset (oversampled)
    if spot_check_dir and Path(spot_check_dir).exists():
        for _ in range(spot_check_oversample):
            sc_ds = NordicOCRDataset(
                spot_check_dir,
                tokenizer=tokenizer,
                img_size=img_size,
                max_label_len=max_label_len,
                augment=True,
            )
            datasets.append(sc_ds)
        print(f"Spot-check: {len(sc_ds)} samples x {spot_check_oversample} = {len(sc_ds) * spot_check_oversample}")

    combined = ConcatDataset(datasets)
    print(f"Total training samples: {len(combined)}")

    return combined


def build_val_dataset(val_dir, tokenizer=None, img_size=(32, 384), max_label_len=128):
    """Build validation dataset (no augmentation)."""
    tokenizer = tokenizer or NordicTokenizer()
    return NordicOCRDataset(
        val_dir,
        tokenizer=tokenizer,
        img_size=img_size,
        max_label_len=max_label_len,
        augment=False,
    )
