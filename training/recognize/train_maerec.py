"""
MAERec fine-tuning for Nordic text recognition.

MAERec (Masked Autoencoder for Scene Text Recognition) uses MAE pre-training
to learn strong visual features for text recognition. It may outperform PARSeq
on degraded/noisy scans where visual feature quality is the bottleneck.

Trade-off vs PARSeq:
  + Stronger visual features from MAE pre-training
  + Better on degraded/noisy text (scans with artifacts)
  - Autoregressive decoding (slower inference, ~3-5x vs PARSeq)
  - Larger model (~50M params vs PARSeq's ~20M)

Use case: If PARSeq's accuracy on degraded scans is insufficient despite
fine-tuning, try MAERec as an alternative.

Usage:
    python train_maerec.py --config config.yaml
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAERec for Nordic OCR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="output/recognize_maerec")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"MAERec Training")
    print(f"  Config: {args.config}")
    print(f"  GPUs: {args.gpus}")

    # Training pipeline:
    # 1. Pre-training phase (MAE):
    #    - Mask 75% of image patches
    #    - Train ViT encoder to reconstruct masked patches
    #    - Use synthetic + real text images (no labels needed)
    #    - 100 epochs, lr=1.5e-4
    #
    # 2. Fine-tuning phase (recognition):
    #    - Add CTC or attention decoder on top of MAE encoder
    #    - Fine-tune on labeled Nordic text line crops
    #    - Same dataset as PARSeq training
    #    - 20 epochs, lr=1e-4
    #
    # 3. Export: PyTorch → ONNX → TensorRT
    #    NOTE: Autoregressive decoding means the ONNX graph has a loop.
    #    Use TensorRT's loop support or export encoder-only and decode on CPU.

    print("Training not implemented — use mmocr MAERec config.")
    print("See: https://github.com/open-mmlab/mmocr")


if __name__ == "__main__":
    main()
