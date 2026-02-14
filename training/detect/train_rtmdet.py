"""
RTMDet fine-tuning for Nordic document text detection.

RTMDet (Real-Time Models for Object Detection) is MMLAB's latest single-stage
detector. It uses a CSPNeXt backbone with large-kernel depthwise convolutions
and a shared detection head — potentially faster than DBNet++ at comparable
accuracy on document text detection.

Usage:
    python train_rtmdet.py --config config.yaml

Benchmark against DBNet++ (train.py) on your Nordic document test set before
switching. RTMDet is proven for general object detection but less tested on
dense document text specifically.

Key differences from DBNet++:
  - Anchor-free, single-stage (vs DBNet++'s segmentation approach)
  - Uses SimOTA label assignment (dynamic, per-image)
  - CSPNeXt backbone instead of ConvNeXt-T
  - Outputs bounding boxes directly (no differentiable binarization)
  - May need NMS post-processing (unlike DBNet++)

Export: PyTorch → ONNX (opset 17) → trtexec --fp8 (Blackwell) or --fp16 (Ada)
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train RTMDet for Nordic OCR")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="output/detect_rtmdet")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"RTMDet Training")
    print(f"  Config: {args.config}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Output: {args.output_dir}")

    # Training pipeline:
    # 1. Load RTMDet-S or RTMDet-M with CSPNeXt backbone (COCO pretrained)
    # 2. Configure for text detection:
    #    - Single class: "text"
    #    - Input: 1024x1024 (match DBNet++ for fair comparison)
    #    - Anchor-free detection head
    # 3. Data: Same Nordic document dataset as DBNet++
    # 4. Train with:
    #    - Loss: Quality Focal Loss + GIoU Loss
    #    - Optimizer: AdamW, lr=4e-3 (RTMDet uses higher lr)
    #    - Schedule: cosine, 300 epochs (RTMDet needs more epochs)
    #    - Batch size: 16 per GPU
    #    - EMA with decay 0.9998
    # 5. Export to ONNX
    #    - NOTE: RTMDet outputs may include NMS — either export with
    #      NMS as part of the ONNX graph, or do NMS in a CUDA kernel
    #      post-TensorRT (avoids CPU round-trip)

    print("Training not implemented — use mmdetection with RTMDet config.")
    print("See: https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet")


if __name__ == "__main__":
    main()
