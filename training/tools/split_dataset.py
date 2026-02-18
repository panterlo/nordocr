"""
Split a training dataset into train/val/test subsets.

Reads a labels.tsv + images/ directory and creates three subdirectories
with their own labels.tsv files.

Usage:
    python split_dataset.py \
        --input D:/TrainingData/auto_labeled \
        --output D:/TrainingData/splits \
        --val-pct 5 --test-pct 5
"""

import argparse
import csv
import os
import random
import shutil
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Input directory (images/ + labels.tsv)")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--val-pct", type=float, default=5.0, help="Validation percentage")
    parser.add_argument("--test-pct", type=float, default=5.0, help="Test percentage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of symlink (use on Windows)")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Read all samples using plain tab-split (csv.DictReader chokes on embedded quotes)
    samples = []
    with open(input_dir / "labels.tsv", "r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\n")
        fieldnames = header_line.split("\t")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) == len(fieldnames):
                samples.append(dict(zip(fieldnames, fields)))

    print(f"Total samples: {len(samples)}")

    # Shuffle and split
    random.shuffle(samples)
    n = len(samples)
    val_size = int(n * args.val_pct / 100)
    test_size = int(n * args.test_pct / 100)
    train_size = n - val_size - test_size

    splits = {
        "train": samples[:train_size],
        "val": samples[train_size:train_size + val_size],
        "test": samples[train_size + val_size:],
    }

    for split_name, split_samples in splits.items():
        split_dir = output_dir / split_name
        img_dir = split_dir / "images"
        os.makedirs(img_dir, exist_ok=True)

        # Copy/link images
        for sample in split_samples:
            src = input_dir / "images" / sample["filename"]
            dst = img_dir / sample["filename"]
            if not dst.exists():
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        # Symlinks may fail on Windows without admin
                        shutil.copy2(src, dst)

        # Write labels.tsv (plain tab-join to avoid csv quoting issues)
        with open(split_dir / "labels.tsv", "w", encoding="utf-8") as f:
            f.write("\t".join(fieldnames) + "\n")
            for sample in split_samples:
                f.write("\t".join(sample[k] for k in fieldnames) + "\n")

        print(f"  {split_name}: {len(split_samples)} samples → {split_dir}")


if __name__ == "__main__":
    main()
