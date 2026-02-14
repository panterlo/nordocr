"""
CLIP4STR fine-tuning for Nordic text recognition.

CLIP4STR leverages CLIP's vision-language pre-training for robust character
recognition. It has the strongest zero-shot generalization to unseen fonts
and styles, making it valuable for documents with unusual typography.

Trade-off vs PARSeq:
  + Best generalization to unseen fonts/styles (CLIP's visual backbone)
  + Strong on mixed-language text (CLIP trained on multilingual data)
  + Cross-attention between visual and language features
  - Much larger model (~100M+ params vs PARSeq's ~20M)
  - Slower inference (~5-10x vs PARSeq)
  - Heavier memory footprint

Use case: Documents with diverse/unusual fonts where PARSeq struggles,
or when you need robust zero-shot handling of rare character combinations.
For standard Nordic documents with consistent typography, PARSeq is preferred.

Usage:
    python train_clip4str.py --config config.yaml
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIP4STR for Nordic OCR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="output/recognize_clip4str")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"CLIP4STR Training")
    print(f"  Config: {args.config}")
    print(f"  GPUs: {args.gpus}")

    # Training pipeline:
    # 1. Load CLIP ViT-B/16 as visual encoder (frozen initially)
    # 2. Add text recognition decoder (cross-attention to CLIP features)
    # 3. Replace CLIP tokenizer with Nordic charset output head
    # 4. Stage 1 — freeze CLIP, train decoder only:
    #    - 5 epochs, lr=1e-3
    # 5. Stage 2 — unfreeze CLIP, fine-tune end-to-end:
    #    - 15 epochs, lr=1e-5 (low lr to preserve CLIP features)
    #    - Same Nordic dataset as PARSeq
    # 6. Export: PyTorch → ONNX → TensorRT
    #    NOTE: Large model — FP8 quantization strongly recommended on Blackwell.
    #    FP4 may cause significant accuracy loss due to CLIP's sensitivity.

    print("Training not implemented.")
    print("See: https://github.com/VamosC/CLIP4STR")


if __name__ == "__main__":
    main()
