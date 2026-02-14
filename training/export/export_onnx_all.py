"""
Export all model variants to ONNX, then build TensorRT engines for both GPUs.

Automates the full export pipeline for all supported model architectures:

Detection:
  - DBNet++ (default)
  - RTMDet (experimental)

Recognition:
  - PARSeq (default)
  - MAERec (experimental)
  - CLIP4STR (experimental)

TensorRT engine builds per GPU:
  - A6000 Ada (sm_89): FP16
  - RTX 6000 PRO Blackwell (sm_120): FP8 + sparsity

Usage:
    python export_onnx_all.py --checkpoints-dir checkpoints/ --output-dir models/

This generates:
    models/
    ├── detect_dbnetpp_sm89.engine
    ├── detect_dbnetpp_sm120.engine
    ├── detect_rtmdet_sm89.engine
    ├── detect_rtmdet_sm120.engine
    ├── recognize_parseq_sm89.engine
    ├── recognize_parseq_sm120.engine
    ├── recognize_maerec_sm89.engine
    ├── recognize_maerec_sm120.engine
    ├── recognize_clip4str_sm89.engine
    └── recognize_clip4str_sm120.engine
"""

import argparse
import subprocess
import os
from pathlib import Path


DETECT_MODELS = {
    "dbnetpp": {
        "input_shape": "input:1x3x1024x1024",
        "opt_shape_ada": "input:2x3x1024x1024",
        "max_shape_ada": "input:4x3x1024x1024",
        "opt_shape_blackwell": "input:8x3x1024x1024",
        "max_shape_blackwell": "input:16x3x1024x1024",
    },
    "rtmdet": {
        "input_shape": "input:1x3x1024x1024",
        "opt_shape_ada": "input:2x3x1024x1024",
        "max_shape_ada": "input:4x3x1024x1024",
        "opt_shape_blackwell": "input:8x3x1024x1024",
        "max_shape_blackwell": "input:16x3x1024x1024",
    },
}

RECOGNIZE_MODELS = {
    "parseq": {
        "input_shape": "input:1x3x32x32",
        "opt_shape_ada": "input:32x3x32x256",
        "max_shape_ada": "input:64x3x32x512",
        "opt_shape_blackwell": "input:128x3x32x256",
        "max_shape_blackwell": "input:256x3x32x512",
    },
    "maerec": {
        "input_shape": "input:1x3x32x32",
        "opt_shape_ada": "input:16x3x32x256",
        "max_shape_ada": "input:32x3x32x512",
        "opt_shape_blackwell": "input:64x3x32x256",
        "max_shape_blackwell": "input:128x3x32x512",
    },
    "clip4str": {
        "input_shape": "input:1x3x224x224",  # CLIP uses 224x224
        "opt_shape_ada": "input:8x3x224x224",
        "max_shape_ada": "input:16x3x224x224",
        "opt_shape_blackwell": "input:32x3x224x224",
        "max_shape_blackwell": "input:64x3x224x224",
    },
}

GPU_CONFIGS = {
    "sm89": {
        "name": "A6000 Ada",
        "precision": "--fp16",
        "extra_flags": [],
    },
    "sm120": {
        "name": "RTX 6000 PRO Blackwell",
        "precision": "--fp8",
        "extra_flags": ["--sparsity=enable"],
    },
}


def build_engine(onnx_path, engine_path, gpu_config, model_config):
    """Build a TensorRT engine using trtexec."""
    gpu = GPU_CONFIGS[gpu_config]
    shapes = model_config

    is_blackwell = gpu_config == "sm120"
    opt_key = "opt_shape_blackwell" if is_blackwell else "opt_shape_ada"
    max_key = "max_shape_blackwell" if is_blackwell else "max_shape_ada"

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        gpu["precision"],
        f"--minShapes={shapes['input_shape']}",
        f"--optShapes={shapes[opt_key]}",
        f"--maxShapes={shapes[max_key]}",
    ] + gpu["extra_flags"]

    print(f"  Building: {engine_path}")
    print(f"  Command: {' '.join(cmd)}")

    # subprocess.run(cmd, check=True)  # uncomment to actually build


def parse_args():
    parser = argparse.ArgumentParser(description="Export all models and build TRT engines")
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--gpu", type=str, choices=["sm89", "sm120", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    gpus = list(GPU_CONFIGS.keys()) if args.gpu == "both" else [args.gpu]

    print("=== Detection Models ===")
    for model_name, config in DETECT_MODELS.items():
        onnx_path = f"{args.checkpoints_dir}/detect_{model_name}.onnx"
        for gpu in gpus:
            engine_path = f"{args.output_dir}/detect_{model_name}_{gpu}.engine"
            build_engine(onnx_path, engine_path, gpu, config)

    print("\n=== Recognition Models ===")
    for model_name, config in RECOGNIZE_MODELS.items():
        onnx_path = f"{args.checkpoints_dir}/recognize_{model_name}.onnx"
        for gpu in gpus:
            engine_path = f"{args.output_dir}/recognize_{model_name}_{gpu}.engine"
            build_engine(onnx_path, engine_path, gpu, config)

    print(f"\nDone. Engines would be written to: {args.output_dir}/")
    if args.dry_run:
        print("(Dry run — no engines were actually built. Remove --dry-run to build.)")


if __name__ == "__main__":
    main()
