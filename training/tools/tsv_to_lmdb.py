"""
Convert TSV + images dataset to LMDB format for OpenOCR / SVTRv2 training.

Reads our standard format:
    data_dir/
        images/000001.png, ...
        labels.tsv  (filename\ttext\tconfidence\tsource_fid)

Produces LMDB with keys:
    num-samples       -> total count
    image-000000001   -> raw PNG/JPEG bytes
    label-000000001   -> UTF-8 text string

Usage:
    python tsv_to_lmdb.py \
        --input D:/TrainingData/splits/train \
        --output D:/TrainingData/lmdb/train

    # Optionally filter by charset:
    python tsv_to_lmdb.py \
        --input D:/TrainingData/splits/train \
        --output D:/TrainingData/lmdb/train \
        --charset-file ../recognize/nordic_dict.txt
"""

import argparse
import os
import sys
from pathlib import Path

import lmdb


def parse_args():
    parser = argparse.ArgumentParser(description="Convert TSV+images to LMDB")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory (images/ + labels.tsv)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output LMDB directory")
    parser.add_argument("--charset-file", type=str, default=None,
                        help="Character dictionary file (one char per line). "
                             "Samples with out-of-charset chars are filtered.")
    parser.add_argument("--max-label-len", type=int, default=100,
                        help="Maximum label length (samples exceeding this are skipped). "
                             "100 covers full-width lines at 1792px.")
    parser.add_argument("--map-size", type=int, default=50,
                        help="LMDB map size in GB (default: 50)")
    return parser.parse_args()


def load_charset(charset_file):
    """Load character set from a dictionary file (one char per line)."""
    chars = set()
    with open(charset_file, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch:
                chars.add(ch)
    return chars


def main():
    args = parse_args()
    input_dir = Path(args.input)
    img_dir = input_dir / "images"
    tsv_path = input_dir / "labels.tsv"

    if not tsv_path.exists():
        print(f"ERROR: {tsv_path} not found")
        sys.exit(1)

    # Optional charset filter
    charset = None
    if args.charset_file:
        charset = load_charset(args.charset_file)
        print(f"Charset filter: {len(charset)} characters from {args.charset_file}")

    # Read samples from TSV
    samples = []
    skipped_charset = 0
    skipped_length = 0
    skipped_missing = 0

    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        fn_idx = header.index("filename")
        txt_idx = header.index("text") if "text" in header else header.index("verified_text")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(fn_idx, txt_idx):
                continue
            filename = parts[fn_idx]
            text = parts[txt_idx]

            # Check image exists
            img_path = img_dir / filename
            if not img_path.exists():
                skipped_missing += 1
                continue

            # Filter by charset
            if charset is not None:
                filtered = "".join(c for c in text if c in charset)
                if filtered != text:
                    skipped_charset += 1
                    continue
                text = filtered

            # Filter by length
            if len(text) > args.max_label_len or len(text) == 0:
                skipped_length += 1
                continue

            samples.append((str(img_path), text))

    print(f"Samples to write: {len(samples)}")
    if skipped_missing:
        print(f"  Skipped (missing image): {skipped_missing}")
    if skipped_charset:
        print(f"  Skipped (out-of-charset): {skipped_charset}")
    if skipped_length:
        print(f"  Skipped (too long/empty): {skipped_length}")

    if not samples:
        print("ERROR: No valid samples found")
        sys.exit(1)

    # Create LMDB
    os.makedirs(args.output, exist_ok=True)
    map_size = args.map_size * (1024 ** 3)  # GB to bytes

    env = lmdb.open(args.output, map_size=map_size)
    cache = {}
    count = 0

    for i, (img_path, label) in enumerate(samples):
        # Read raw image bytes (no decoding — preserves original format)
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # 1-indexed, zero-padded to 9 digits
        key_idx = f"{i + 1:09d}"
        cache[f"image-{key_idx}".encode()] = img_bytes
        cache[f"label-{key_idx}".encode()] = label.encode("utf-8")
        count += 1

        # Write in batches of 1000
        if count % 1000 == 0:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
            if count % 10000 == 0:
                print(f"  Written {count}/{len(samples)} samples...")

    # Write remaining
    cache[b"num-samples"] = str(count).encode()
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

    env.close()
    print(f"\nLMDB created: {args.output}")
    print(f"  Total samples: {count}")


if __name__ == "__main__":
    main()
