"""
PARSeq fine-tuning for Nordic text recognition.

Usage:
    python train.py --config config.yaml

Base model: PARSeq-S (ViT-Small, ~20M params)
Extended charset: ASCII + å ä ö ø æ Å Ä Ö Ø Æ ð þ Ð Þ ü Ü + punctuation
Fine-tune on Nordic synthetic + real scanned text line crops.
"""

import argparse
from pathlib import Path


NORDIC_CHARSET = (
    # Special tokens
    "[PAD][UNK][EOS]"
    # Digits
    "0123456789"
    # ASCII letters
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    # Nordic uppercase
    "ÅÄÖØÆÐÞÜ"
    # Nordic lowercase
    "åäöøæðþü"
    # Punctuation
    " !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    # Extra
    "§°€£«»–—''"""
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train PARSeq for Nordic OCR")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--output-dir", type=str, default="output/recognize")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"PARSeq Training")
    print(f"  Config: {args.config}")
    print(f"  Charset size: {len(NORDIC_CHARSET)}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Output: {args.output_dir}")

    # Training pipeline:
    # 1. Load PARSeq-S base model (pretrained on MJSynth + SynthText)
    # 2. Replace output head for Nordic charset
    # 3. Datasets:
    #    a. MJSynth + SynthText (English synthetic — transfer learning)
    #    b. Nordic synthetic:
    #       - Swedish: SUC 3.0, Talbanken, Swedish Wikipedia text
    #       - Norwegian: NCC (NB AI Lab), Norwegian Wikipedia
    #       - Danish: Danish Gigaword, Danish Wikipedia
    #       - Finnish: FinnSentiment, Finnish Wikipedia
    #       Rendered with Nordic-style fonts onto scanned paper backgrounds
    #    c. Real scanned text line crops (as available)
    # 4. Augmentation:
    #    - Geometric: rotation ±5°, perspective, scaling
    #    - Photometric: blur, noise (Gaussian, salt-pepper), erosion, dilation
    #    - Diacritical-aware: ensure balanced å/ä/ö/ø/æ in training batches
    # 5. Training:
    #    - PARSeq permutation training (all 6 autoregressive orders)
    #    - Optimizer: AdamW, lr=7e-4, weight_decay=0.05
    #    - Schedule: 1-cycle, 20 epochs
    #    - Batch size: 384 per GPU
    # 6. Evaluation:
    #    - Standard: CER, WER on combined test set
    #    - Nordic-specific: diacritical accuracy (å vs a, ö vs o, etc.)

    print("Training not implemented — use the PARSeq codebase (baudm/parseq).")
    print("Key: fine-tune with --charset from NORDIC_CHARSET above.")


if __name__ == "__main__":
    main()
