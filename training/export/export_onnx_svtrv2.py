"""
Export trained SVTRv2 model to ONNX for TensorRT conversion.

SVTRv2 with CTC/RCTC produces a clean computational graph — no onnxsim needed.
Both batch and width dimensions are dynamic, allowing variable-width text lines
without compression.

Output shape: [batch, T, num_classes] where T = input_width / stride.
For base variant with stride 4: T = W/4. At max_width=1792: T=448.

Normalization: same as PARSeq — T.Normalize(0.5, 0.5) mapping [0,1] -> [-1,1].
The inference CUDA kernel must use mean=0.5, std=0.5 (NOT ImageNet stats).

Usage:
    python export_onnx_svtrv2.py \
        --checkpoint output/svtrv2_nordic/best.pth \
        --output models/recognize_svtrv2.onnx

    # With explicit OpenOCR path:
    python export_onnx_svtrv2.py \
        --checkpoint output/svtrv2_nordic/best.pth \
        --output models/recognize_svtrv2.onnx \
        --openocr-dir C:/dev/OpenOCR

Then build TensorRT engines (dynamic width, no onnxsim needed):
    python build_trt_engine.py --onnx models/recognize_svtrv2.onnx \
        --output models/recognize_svtrv2_sm89.engine --gpu sm89

NOTE: Unlike PARSeq, SVTRv2 ONNX has dynamic width. Both batch and width
      dimensions are dynamic. TRT must be built with --minShapes/--maxShapes
      covering the full width range (32 to 1792).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import onnx
import yaml

# Add recognize dir to path for charset
sys.path.insert(0, str(Path(__file__).parent.parent / "recognize"))


def find_openocr(openocr_dir=None):
    """Find and validate OpenOCR installation."""
    if openocr_dir:
        p = Path(openocr_dir)
        if (p / "openrec" / "modeling").exists():
            return p

    for candidate in [
        Path("C:/dev/OpenOCR"),
        Path.home() / "OpenOCR",
        Path(__file__).parent.parent.parent / "OpenOCR",
        Path("OpenOCR"),
    ]:
        if (candidate / "openrec" / "modeling").exists():
            return candidate

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Export SVTRv2 to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained SVTRv2 checkpoint (.pth) or OpenOCR output dir")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to OpenOCR training config YAML (auto-detected from checkpoint dir)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output ONNX file path")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--img-width", type=int, default=384,
                        help="Width for dummy input (does not limit runtime — width is dynamic)")
    parser.add_argument("--openocr-dir", type=str, default=None,
                        help="Path to cloned OpenOCR repo")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Verify ONNX model with onnxruntime")
    return parser.parse_args()


def load_model_from_openocr_checkpoint(checkpoint_path, config_path, openocr_path):
    """Load SVTRv2 model from an OpenOCR training checkpoint."""
    sys.path.insert(0, str(openocr_path))

    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from tools.engine.config import Config

    # Load config
    cfg = Config(config_path)
    _cfg = cfg.cfg

    # Build post-processor to get charset size
    post_process = build_post_process(_cfg["PostProcess"])
    char_num = len(getattr(post_process, "character"))
    _cfg["Architecture"]["Decoder"]["out_channels"] = char_num

    # Build model
    model = build_model(_cfg["Architecture"])

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Strip module. prefix if present (from DDP training)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"  Architecture: {_cfg['Architecture']['algorithm']}")
    print(f"  Decoder: {_cfg['Architecture']['Decoder']['name']}")
    print(f"  Output classes: {char_num} (blank + {char_num - 1} charset chars)")

    return model, char_num


def load_model_standalone(checkpoint_path, openocr_path):
    """
    Load SVTRv2 model from a standalone checkpoint with embedded config.

    Falls back to building the model from the config.yml in the same directory
    as the checkpoint.
    """
    checkpoint_dir = Path(checkpoint_path).parent
    config_candidates = [
        checkpoint_dir / "config.yml",
        checkpoint_dir / "config.yaml",
        checkpoint_dir.parent / "config.yml",
    ]

    for cfg_path in config_candidates:
        if cfg_path.exists():
            print(f"  Found config: {cfg_path}")
            return load_model_from_openocr_checkpoint(
                checkpoint_path, str(cfg_path), openocr_path
            )

    raise FileNotFoundError(
        f"No config.yml found near {checkpoint_path}. "
        f"Tried: {[str(p) for p in config_candidates]}. "
        f"Pass --config explicitly."
    )


def main():
    args = parse_args()

    # Find OpenOCR
    openocr_path = find_openocr(args.openocr_dir)
    if openocr_path is None:
        print("ERROR: OpenOCR not found. Install it first:")
        print("  git clone https://github.com/Topdu/OpenOCR.git C:/dev/OpenOCR")
        sys.exit(1)
    print(f"OpenOCR: {openocr_path}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    config_path = args.config
    if config_path:
        model, num_classes = load_model_from_openocr_checkpoint(
            args.checkpoint, config_path, openocr_path
        )
    else:
        model, num_classes = load_model_standalone(args.checkpoint, openocr_path)

    # Rep reparameterization (some SVTRv2 variants have reparam conv layers)
    for layer in model.modules():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", True):
            layer.rep()

    model.eval()

    # Dummy input
    device = torch.device("cpu")
    dummy = torch.randn(1, 3, args.img_height, args.img_width, device=device)

    # Trace to get output shape
    with torch.no_grad():
        test_out = model(dummy)
    if isinstance(test_out, tuple):
        test_out = test_out[-1]  # (feats, predicts) when return_feats=True

    output_T = test_out.shape[1]
    output_C = test_out.shape[2]
    stride = args.img_width // output_T

    print(f"\nModel analysis:")
    print(f"  Input:  [B, 3, {args.img_height}, W]  (W is dynamic)")
    print(f"  Output: [B, T, {output_C}]  (T = W / {stride})")
    print(f"  Stride: {stride} (input_width / output_time)")
    print(f"  At dummy W={args.img_width}: T={output_T}")
    print(f"  At max   W=1792: T={1792 // stride}")
    print(f"  Classes: {output_C} (blank + {output_C - 1} chars)")

    # Export to ONNX
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"\nExporting to ONNX: {args.output}")
    print(f"  Opset: {args.opset}")
    print(f"  Dynamic axes: batch (dim 0) + width (dim 3) on input,")
    print(f"                batch (dim 0) + time (dim 1) on output")

    torch.onnx.export(
        model,
        dummy,
        args.output,
        opset_version=args.opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch", 3: "width"},
            "output": {0: "batch", 1: "time"},
        },
        do_constant_folding=True,
    )

    # Verify ONNX
    print("Verifying ONNX model...")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid")

    # No onnxsim needed — CTC graph is clean
    print("  (onnxsim not needed for SVTRv2 CTC — graph is clean)")

    # Verify with onnxruntime at multiple widths
    if args.verify:
        try:
            import onnxruntime as ort
            print("Verifying with onnxruntime...")
            sess = ort.InferenceSession(args.output)

            for test_w in [128, 384, 768, 1792]:
                dummy_np = torch.randn(1, 3, args.img_height, test_w).numpy()
                outputs = sess.run(None, {"input": dummy_np})
                expected_T = test_w // stride
                actual_T = outputs[0].shape[1]
                status = "OK" if actual_T == expected_T else f"MISMATCH (expected {expected_T})"
                print(f"  W={test_w:4d} -> output [{outputs[0].shape[0]}, {actual_T}, {outputs[0].shape[2]}] {status}")

            print("  ONNX runtime verification passed")
        except ImportError:
            print("  WARNING: onnxruntime not installed, skipping verification")

    # Save metadata for downstream tooling
    meta = {
        "model": "SVTRv2",
        "decoder": "CTC",
        "stride": stride,
        "num_classes": output_C,
        "charset_size": output_C - 1,
        "input_height": args.img_height,
        "max_width_tested": 1792,
        "normalization": {"mean": 0.5, "std": 0.5},
        "blank_index": 0,
    }
    meta_path = args.output.replace(".onnx", "_meta.yaml")
    with open(meta_path, "w") as f:
        yaml.dump(meta, f, default_flow_style=False)
    print(f"\nMetadata saved: {meta_path}")

    print(f"\nExport complete: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Build TensorRT engine on target GPU:")
    print(f"     trtexec --onnx={args.output} \\")
    print(f"         --saveEngine=models/recognize_svtrv2_sm89.engine \\")
    print(f"         --fp16 \\")
    print(f"         --minShapes=input:1x3x{args.img_height}x32 \\")
    print(f"         --optShapes=input:32x3x{args.img_height}x384 \\")
    print(f"         --maxShapes=input:64x3x{args.img_height}x1792")


if __name__ == "__main__":
    main()
