"""
Export trained PyTorch models to ONNX, then build TensorRT engines.

Usage:
    python export_onnx.py --model detect --checkpoint path/to/best.pth --output models/detect.onnx
    python export_onnx.py --model recognize --checkpoint path/to/best.pth --output models/recognize.onnx

After ONNX export, build TensorRT engines with:
    trtexec --onnx=models/detect.onnx --saveEngine=models/detect.engine \
            --fp8 --sparsity=enable --minShapes=input:1x3x1024x1024 \
            --optShapes=input:4x3x1024x1024 --maxShapes=input:8x3x1024x1024

    trtexec --onnx=models/recognize.onnx --saveEngine=models/recognize.engine \
            --fp8 --minShapes=input:1x3x32x32 \
            --optShapes=input:64x3x32x256 --maxShapes=input:128x3x32x512
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["detect", "recognize"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true", default=True)
    return parser.parse_args()


def export_detection(args):
    """Export DBNet++ to ONNX."""
    print(f"Exporting detection model: {args.checkpoint} -> {args.output}")

    # In production:
    #   import torch
    #   model = load_dbnet_pp(args.checkpoint)
    #   model.eval()
    #   dummy_input = torch.randn(1, 3, 1024, 1024)
    #   dynamic_axes = {"input": {0: "batch"}, "prob_map": {0: "batch"}, "thresh_map": {0: "batch"}}
    #   torch.onnx.export(
    #       model, dummy_input, args.output,
    #       opset_version=args.opset,
    #       input_names=["input"],
    #       output_names=["prob_map", "thresh_map"],
    #       dynamic_axes=dynamic_axes if args.dynamic_batch else None,
    #   )

    print("ONNX export not implemented — requires PyTorch model definition.")


def export_recognition(args):
    """Export PARSeq to ONNX."""
    print(f"Exporting recognition model: {args.checkpoint} -> {args.output}")

    # In production:
    #   import torch
    #   from parseq import PARSeq
    #   model = PARSeq.load_from_checkpoint(args.checkpoint)
    #   model.eval()
    #   # PARSeq input: [batch, 3, 32, W] where W is variable
    #   dummy_input = torch.randn(1, 3, 32, 128)
    #   dynamic_axes = {"input": {0: "batch", 3: "width"}, "logits": {0: "batch"}}
    #   torch.onnx.export(
    #       model.encoder_decoder, dummy_input, args.output,
    #       opset_version=args.opset,
    #       input_names=["input"],
    #       output_names=["logits"],
    #       dynamic_axes=dynamic_axes if args.dynamic_batch else None,
    #   )

    print("ONNX export not implemented — requires PARSeq model definition.")


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.model == "detect":
        export_detection(args)
    elif args.model == "recognize":
        export_recognition(args)


if __name__ == "__main__":
    main()
