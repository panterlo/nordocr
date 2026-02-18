"""
Build TensorRT engine from ONNX model using the TensorRT Python API.

Replaces trtexec CLI usage. Handles ONNX simplification (via onnxoptimizer
when onnxsim is unavailable) and builds GPU-specific engines.

Usage (RTX A6000 Ampere / SM_86 — dev machine):
    python build_trt_engine.py \
        --onnx models/recognize_parseq.onnx \
        --output models/recognize_sm86.engine \
        --gpu sm86

Usage (Ada Lovelace / SM_89):
    python build_trt_engine.py \
        --onnx models/recognize_parseq.onnx \
        --output models/recognize_sm89.engine \
        --gpu sm89

Usage (RTX 6000 PRO Blackwell / SM_120):
    python build_trt_engine.py \
        --onnx models/recognize_parseq.onnx \
        --output models/recognize_sm120.engine \
        --gpu sm120
"""

import argparse
import os
import sys
import time

import onnx
import onnx.shape_inference
import onnxoptimizer
import tensorrt as trt


# Batch size profiles per GPU target (applied to dynamic batch dimension)
GPU_PROFILES = {
    "sm86": {
        # RTX A6000 Ampere 48GB (dev machine)
        "recognize_batch": {"min": 1, "opt": 32, "max": 64},
        "detect_batch": {"min": 1, "opt": 2, "max": 4},
        "precision": "fp16",
    },
    "sm89": {
        # Ada Lovelace (RTX 4090, L40, etc.)
        "recognize_batch": {"min": 1, "opt": 32, "max": 64},
        "detect_batch": {"min": 1, "opt": 2, "max": 4},
        "precision": "fp16",
    },
    "sm120": {
        # RTX 6000 PRO Blackwell 96GB (training machine)
        "recognize_batch": {"min": 1, "opt": 128, "max": 256},
        "detect_batch": {"min": 1, "opt": 8, "max": 16},
        "precision": "fp8",
        "sparsity": True,
    },
}


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def simplify_onnx(onnx_path):
    """Simplify ONNX model using onnxoptimizer (fallback when onnxsim unavailable).

    PARSeq's attention graph contains complex ops that TRT may not handle well.
    Simplification folds constants, fuses ops, and removes dead nodes.
    """
    print(f"Loading ONNX model: {onnx_path}")
    model = onnx.load(onnx_path)

    # Run shape inference first (needed for optimization passes)
    print("  Running shape inference...")
    model = onnx.shape_inference.infer_shapes(model)

    # Apply optimization passes
    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_pad_into_pool",
        "fuse_transpose_into_gemm",
        "nop",
    ]

    print(f"  Applying {len(passes)} optimization passes...")
    optimized = onnxoptimizer.optimize(model, passes)

    # Run shape inference again after optimization
    optimized = onnx.shape_inference.infer_shapes(optimized)

    # Validate
    onnx.checker.check_model(optimized)

    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    onnx.save(optimized, simplified_path)
    print(f"  Simplified model saved: {simplified_path}")

    return simplified_path


def build_engine(onnx_path, engine_path, gpu_profile, model_type="recognize"):
    """Build a TensorRT engine from an ONNX model.

    Auto-detects dynamic vs fixed dimensions from the ONNX model and applies
    batch size profiles only to dynamic dimensions.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config = builder.create_builder_config()

    # Parse ONNX
    print(f"\nParsing ONNX model: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            sys.exit(1)

    print(f"  Network: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  Input '{inp.name}': {inp.shape} ({inp.dtype})")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  Output '{out.name}': {out.shape} ({out.dtype})")

    # Memory — allow TRT to use up to 4 GB workspace
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    # Precision
    precision = gpu_profile.get("precision", "fp16")
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  Enabled FP16 precision")
        else:
            print("  WARNING: GPU does not support fast FP16")
    elif precision == "fp8":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if hasattr(trt.BuilderFlag, "FP8"):
            config.set_flag(trt.BuilderFlag.FP8)
            print("  Enabled FP8 precision")
        else:
            print("  WARNING: TRT version does not support FP8, falling back to FP16")

    # Sparsity
    if gpu_profile.get("sparsity"):
        if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            print("  Enabled structured sparsity")

    # Build optimization profile from ONNX model's actual shapes
    # Only dynamic dimensions (marked as -1) get min/opt/max ranges
    batch_key = f"{model_type}_batch"
    batch_profile = gpu_profile[batch_key]

    opt_profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = inp.shape
        has_dynamic = any(d == -1 for d in shape)

        if has_dynamic:
            # Build min/opt/max by replacing -1 dims with batch profile values
            min_shape = tuple(batch_profile["min"] if d == -1 else d for d in shape)
            opt_shape = tuple(batch_profile["opt"] if d == -1 else d for d in shape)
            max_shape = tuple(batch_profile["max"] if d == -1 else d for d in shape)

            print(f"  Dynamic shapes for '{inp.name}':")
            print(f"    ONNX shape: {list(shape)}")
            print(f"    min: {min_shape}")
            print(f"    opt: {opt_shape}")
            print(f"    max: {max_shape}")

            opt_profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        else:
            print(f"  Static shape for '{inp.name}': {list(shape)}")

    config.add_optimization_profile(opt_profile)

    # Build engine
    print(f"\nBuilding TensorRT engine ({precision.upper()})...")
    print("  This may take several minutes...")
    t0 = time.time()

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: Engine build failed!")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"  Engine built in {elapsed:.1f}s")

    # Save
    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Saved: {engine_path} ({size_mb:.1f} MB)")

    return engine_path


def verify_engine(engine_path):
    """Quick verification that the engine can be deserialized."""
    print(f"\nVerifying engine: {engine_path}")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("  ERROR: Failed to deserialize engine!")
        return False

    print(f"  Engine OK: {engine.num_io_tensors} I/O tensors")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        print(f"    {mode.name} '{name}': {list(shape)} ({dtype})")

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--output", type=str, required=True, help="Output engine path")
    parser.add_argument("--gpu", type=str, choices=["sm86", "sm89", "sm120"], default="sm86",
                        help="Target GPU (sm86=A6000 Ampere, sm89=Ada, sm120=Blackwell)")
    parser.add_argument("--model-type", type=str, choices=["recognize", "detect"], default="recognize",
                        help="Model type (affects shape profiles)")
    parser.add_argument("--skip-simplify", action="store_true",
                        help="Skip ONNX simplification (use if already simplified)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip engine verification after build")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"=== TensorRT Engine Builder ===")
    print(f"TensorRT version: {trt.__version__}")
    print(f"Target GPU: {args.gpu}")
    print(f"Model type: {args.model_type}")
    print()

    # Select GPU profile
    profile = GPU_PROFILES[args.gpu]

    # Simplify ONNX
    onnx_path = args.onnx
    if not args.skip_simplify:
        onnx_path = simplify_onnx(args.onnx)
    else:
        print(f"Skipping simplification, using: {onnx_path}")

    # Build engine
    build_engine(onnx_path, args.output, profile, args.model_type)

    # Verify
    if not args.skip_verify:
        if not verify_engine(args.output):
            sys.exit(1)

    print(f"\nDone! Engine ready: {args.output}")


if __name__ == "__main__":
    main()
