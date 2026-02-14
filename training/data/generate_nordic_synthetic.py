"""
Generate synthetic Nordic text line images for PARSeq training.

Renders text from Nordic language corpora onto scanned paper backgrounds
with realistic augmentation (blur, noise, erosion/dilation).

Usage:
    python generate_nordic_synthetic.py \
        --text-dir corpora/ \
        --backgrounds backgrounds/ \
        --output output/synthetic/ \
        --num-samples 1000000
"""

import argparse
import os
import random
from pathlib import Path


# Nordic-specific character distributions for balanced training.
NORDIC_SPECIAL_CHARS = {
    "sv": ["å", "ä", "ö", "Å", "Ä", "Ö"],         # Swedish
    "no": ["ø", "æ", "å", "Ø", "Æ", "Å"],         # Norwegian
    "da": ["ø", "æ", "å", "Ø", "Æ", "Å"],         # Danish
    "fi": ["ä", "ö", "Ä", "Ö"],                     # Finnish
    "is": ["ð", "þ", "æ", "ö", "Ð", "Þ", "Æ", "Ö"],  # Icelandic
}

# Diacritical confusion pairs — ensure these are well-represented.
CONFUSION_PAIRS = [
    ("a", "å"), ("a", "ä"), ("å", "ä"),
    ("o", "ö"), ("o", "ø"), ("ö", "ø"),
    ("A", "Å"), ("A", "Ä"), ("Å", "Ä"),
    ("O", "Ö"), ("O", "Ø"), ("Ö", "Ø"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Nordic synthetic OCR data")
    parser.add_argument("--text-dir", type=str, required=True,
                        help="Directory with text corpora (one .txt per language)")
    parser.add_argument("--backgrounds", type=str, required=True,
                        help="Directory with scanned paper background images")
    parser.add_argument("--fonts-dir", type=str, default="/usr/share/fonts",
                        help="Directory with TTF/OTF fonts")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for synthetic images + labels")
    parser.add_argument("--num-samples", type=int, default=100000)
    parser.add_argument("--height", type=int, default=32,
                        help="Output image height (PARSeq standard)")
    parser.add_argument("--min-width", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    print(f"Generating {args.num_samples} Nordic synthetic text images")
    print(f"  Text corpora: {args.text_dir}")
    print(f"  Backgrounds: {args.backgrounds}")
    print(f"  Output: {args.output}")

    # Generation pipeline (per sample):
    # 1. Sample a random text line from a Nordic corpus
    # 2. Select a random font (prefer Nordic-compatible fonts)
    # 3. Render text onto a random background crop
    # 4. Apply augmentation:
    #    - Blur (Gaussian, motion)
    #    - Noise (Gaussian, salt-pepper)
    #    - Erosion/dilation (morphological)
    #    - Perspective warp (slight)
    #    - Brightness/contrast jitter
    # 5. Resize to fixed height (32px), variable width
    # 6. Save image + label (text string)
    #
    # Ensure balanced representation of Nordic diacriticals:
    # - At least 10% of samples contain å/ä/ö/ø/æ characters
    # - Include confusion-pair samples (e.g., "hår" vs "har")

    print("Generation not implemented — requires Pillow/PIL rendering pipeline.")
    print("See PARSeq codebase for synthetic data generation patterns.")


if __name__ == "__main__":
    main()
