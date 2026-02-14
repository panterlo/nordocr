# nordocr — Development Handoff

## What This Is

A Rust workspace implementing a GPU-resident OCR pipeline for scanned Nordic-language documents. Replaces Tesseract with custom CUDA kernels + TensorRT inference for DBNet++ (detection) and PARSeq (recognition).

### Target GPUs

| Environment | GPU | Architecture | SM | VRAM |
|---|---|---|---|---|
| **Production** | RTX 6000 PRO Blackwell | Blackwell | sm_120 | 96 GB |
| **Development** | A6000 Ada | Ada Lovelace | sm_89 | 48 GB |

CUDA kernels are compiled as **fat binaries** containing native code for both architectures. The CUDA driver selects the right code at load time. TensorRT engines must be built separately per GPU (they are not portable across architectures).

## Current State

**Scaffolding complete, not yet compiled.** All crate structures, types, traits, API surfaces, CUDA kernels, and pipeline wiring are in place. The code was written on a Windows machine without Rust or CUDA — it has never been through `cargo check`.

### What exists and is substantive

- **nordocr-core** — fully implemented types (`BBox`, `TextLine`, `PageResult`, `Polygon`, `FileInput`, etc.), error types, `PipelineStage` trait
- **nordocr-gpu** — `GpuContext`, `GpuMemoryPool` (slab allocator avoiding cudaMalloc during inference), `GpuBuffer<T>` (typed GPU memory wrapper), `StreamPool`
- **nordocr-trt-sys** — bindgen build.rs (gated behind `generated` feature flag) + stub types that compile without TensorRT headers
- **nordocr-trt** — safe wrappers: `TrtRuntime`, `TrtEngine`, `TrtExecutionContext`, `TrtEngineBuilder` (FP8/FP4/sparsity config), `CudaGraph` capture/replay
- **nordocr-preprocess** — 4 CUDA kernels (.cu files with real algorithms) + Rust wrappers + `PreprocessPipeline` orchestrating denoise→deskew→binarize→morphology. **Multi-arch build**: compiles PTX for both sm_89 and sm_120, plus fat binaries. Runtime GPU detection via `GpuArch::detect()`.
- **nordocr-decode** — `ImageDecoder` (nvJPEG path + CPU fallback), `PdfDecoder` (pdfium-render), unified `Decoder`
- **nordocr-detect** — `DetectionEngine` (TensorRT), `DetectionPostprocessor` (prob map → bboxes), `DetectionBatcher`
- **nordocr-recognize** — `RecognitionEngine` (TensorRT), width-sorted `RecognitionBatcher`, `TokenDecoder` (softmax + argmax), full Nordic `charset` with tests
- **nordocr-pipeline** — `OcrPipeline` wiring all stages, `PageScheduler` (round-robin stream assignment), `PipelineConfig` with `a6000_ada()` and `rtx6000_pro_blackwell()` presets
- **nordocr-server** — axum HTTP API (`POST /ocr`, `POST /ocr/json`, `GET /health`), CLI with `serve`/`process`/`batch`/`build-engines` subcommands, Prometheus metrics
- **CUDA kernels** — `binarize.cu` (Sauvola adaptive with integral images), `deskew.cu` (projection profile + affine warp), `denoise.cu` (bilateral, median, gaussian), `morphology.cu` (dilate, erode, CCL, small component removal)
- **Training scripts** — Python stubs for DBNet++ fine-tuning, PARSeq fine-tuning, ONNX export, Nordic synthetic data generation

### What is placeholder / needs real implementation

Every function that actually calls CUDA or TensorRT APIs has a comment block showing the intended implementation but returns a dummy value. Look for patterns like:

```rust
// In production:
//   let module = ctx.device.load_ptx(...);
```

These are the integration points that need real FFI calls once CUDA is available.

## First Steps on the Target Machine

### 1. Install prerequisites

```bash
# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Verify CUDA
nvcc --version          # need CUDA 12.x
nvidia-smi              # confirm GPU visible (A6000 Ada or RTX 6000 PRO Blackwell)

# TensorRT (for full build with generated bindings)
apt install libnvinfer-dev libnvinfer-plugin-dev  # or from NVIDIA .deb
```

### 2. First compilation pass

```bash
cd nordocr

# This will compile everything except TensorRT bindgen (stub mode).
# nvcc compiles .cu → .ptx for sm_89 + sm_120, plus fat binaries.
# If nvcc is not found, build.rs creates stub .ptx files.
cargo check

# Fix whatever comes up — expect mostly minor type mismatches,
# missing trait imports, lifetime issues. The structure is sound
# but hasn't been through the compiler yet.
```

**To compile CUDA kernels for only your dev GPU** (faster builds):

```bash
NORDOCR_CUDA_ARCHS="sm_89" cargo check    # A6000 Ada only
NORDOCR_CUDA_ARCHS="sm_120" cargo check   # RTX 6000 PRO Blackwell only
NORDOCR_CUDA_ARCHS="sm_89,sm_120" cargo check  # both (default)
```

### 3. Known things that will need fixing

| Issue | Where | What to do |
|---|---|---|
| `cudarc` API mismatches | `nordocr-gpu/src/*.rs` | Check cudarc 0.12 docs — method names and generics may differ from what's written |
| `CudaStream` type | `nordocr-gpu/src/stream.rs` | cudarc's stream API may use `CudaStream` differently than assumed; adapt |
| `DevicePtr` trait | `nordocr-gpu/src/buffer.rs`, `memory.rs` | Verify `*slice.device_ptr()` syntax matches cudarc 0.12 |
| `image` crate API | `nordocr-decode/src/image.rs` | `image` 0.25 API for `load_from_memory`, `to_luma8` — should be fine but verify |
| `axum` extractor types | `nordocr-server/src/api.rs` | axum 0.7 multipart — verify `Multipart` import path |
| `metrics` crate API | `nordocr-server/src/api.rs`, `main.rs` | `metrics` 0.23 + `metrics-exporter-prometheus` 0.15 — verify builder API |
| `half::f16` arithmetic | `nordocr-recognize/src/decode.rs` | `f32::from(f16)` should work with `half` 2.x `num-traits` feature |
| `pdfium-render` | `nordocr-decode/src/pdf.rs` | Needs pdfium binary at runtime; the whole module is behind a `Result::Err` stub |

### 4. Wire up real CUDA calls

The highest-value work after getting `cargo check` clean:

1. **nordocr-gpu**: Make `GpuContext::new`, `GpuMemoryPool::alloc/free`, `StreamPool` work with real cudarc calls. This unblocks everything.

2. **nordocr-preprocess**: Load compiled PTX via `cudarc`, launch kernels. The `.cu` files have real algorithms — the Rust wrappers just need real `launch!` calls. The `GpuArch` detection in `gpu_arch.rs` needs the real cudarc device attribute query wired up.

3. **nordocr-trt**: Replace placeholder handles with actual TensorRT C API calls through `nordocr-trt-sys`. Enable the `generated` feature to get real bindgen output:
   ```bash
   cargo build --features nordocr-trt-sys/generated
   ```

4. **nordocr-detect** / **nordocr-recognize**: Once TRT wrappers work, these engines just need real `enqueueV3` calls and buffer binding.

### 5. Get models

TensorRT engines are **GPU-architecture-specific**. You need separate engines for each target GPU.

```bash
# On A6000 Ada (dev):
trtexec --onnx=models/dbnet_pp.onnx \
        --saveEngine=models/detect_sm89.engine \
        --fp16 \
        --minShapes=input:1x3x1024x1024 \
        --optShapes=input:2x3x1024x1024 \
        --maxShapes=input:4x3x1024x1024

trtexec --onnx=models/parseq_s.onnx \
        --saveEngine=models/recognize_sm89.engine \
        --fp16 \
        --minShapes=input:1x3x32x32 \
        --optShapes=input:32x3x32x256 \
        --maxShapes=input:64x3x32x512

# On RTX 6000 PRO Blackwell (prod):
trtexec --onnx=models/dbnet_pp.onnx \
        --saveEngine=models/detect_sm120.engine \
        --fp8 --sparsity=enable \
        --minShapes=input:1x3x1024x1024 \
        --optShapes=input:8x3x1024x1024 \
        --maxShapes=input:16x3x1024x1024

trtexec --onnx=models/parseq_s.onnx \
        --saveEngine=models/recognize_sm120.engine \
        --fp8 \
        --minShapes=input:1x3x32x32 \
        --optShapes=input:128x3x32x256 \
        --maxShapes=input:256x3x32x512
```

Note: FP8 is available on Blackwell (sm_120) but not on Ada (sm_89). Use FP16 on Ada.

Use `PipelineConfig::a6000_ada()` or `PipelineConfig::rtx6000_pro_blackwell()` presets to get the right engine paths and batch sizes automatically.

To get the ONNX models, train in Python first (see `training/` directory) or download pretrained weights and export.

## Architecture Quick Reference

```
Image bytes ──► GPU Decode ──► Preprocess ──► Detect ──► Recognize ──► Text out
  (only CPU→GPU copy)      (custom CUDA)   (TensorRT)  (TensorRT)  (only GPU→CPU copy)
                           denoise          DBNet++     PARSeq
                           deskew           prob map    logits
                           binarize         → bboxes    → tokens
                           morphology                   → text
```

Everything between the two arrows stays in GPU memory. The `GpuMemoryPool` pre-allocates slabs to avoid cudaMalloc during inference.

## Multi-GPU-Architecture Build

```
build.rs (nordocr-preprocess)
  │
  ├── nvcc --fatbin --generate-code=arch=compute_89,code=sm_89
  │                 --generate-code=arch=compute_120,code=sm_120
  │                 --generate-code=arch=compute_89,code=compute_89   ← PTX for JIT
  │                 --generate-code=arch=compute_120,code=compute_120 ← PTX for JIT
  │   → binarize.fatbin  (runs on both GPUs, driver picks best code)
  │
  ├── nvcc --ptx --generate-code=arch=compute_89,code=sm_89
  │   → binarize_sm89.ptx  (Ada-specific, used if fatbin not available)
  │
  └── nvcc --ptx --generate-code=arch=compute_120,code=sm_120
      → binarize_sm120.ptx  (Blackwell-specific)

Runtime (gpu_arch.rs):
  GpuArch::detect() → queries CUDA device compute capability
    sm_89  → load *_sm89.ptx
    sm_120 → load *_sm120.ptx (fallback to sm_89 if unavailable)
```

Override: `NORDOCR_TARGET_ARCH=sm_89` or `NORDOCR_TARGET_ARCH=sm_120`

## Crate Dependency Graph

```
nordocr-core          (no deps — shared types/errors)
  ├── nordocr-gpu     (cudarc — GPU context, memory, streams)
  ├── nordocr-trt-sys (bindgen — raw TensorRT FFI)
  │     └── nordocr-trt (safe TensorRT wrappers)
  ├── nordocr-preprocess (CUDA kernels via cudarc + gpu_arch multi-target)
  ├── nordocr-decode  (image/pdfium — input decoding)
  ├── nordocr-detect  (nordocr-trt — DBNet++ inference)
  ├── nordocr-recognize (nordocr-trt — PARSeq inference)
  ├── nordocr-pipeline (wires everything together)
  └── nordocr-server  (axum HTTP + clap CLI)
```

## File Inventory

| Path | Files | Description |
|---|---|---|
| `Cargo.toml` | 1 | Workspace root with all shared deps |
| `crates/nordocr-core/src/` | 4 | Types, traits, errors |
| `crates/nordocr-gpu/src/` | 5 | GPU context, memory pool, buffers, streams |
| `crates/nordocr-trt-sys/` | 2 | build.rs + stub/generated FFI |
| `crates/nordocr-trt/src/` | 4 | Safe TRT runtime, builder, CUDA graphs |
| `crates/nordocr-preprocess/` | 11 | 4 .cu kernels + build.rs + 6 Rust wrappers (incl. gpu_arch) |
| `crates/nordocr-decode/src/` | 3 | Image + PDF decoding |
| `crates/nordocr-detect/src/` | 4 | Detection engine, postproc, batching |
| `crates/nordocr-recognize/src/` | 5 | Recognition engine, batching, decoding, charset |
| `crates/nordocr-pipeline/src/` | 4 | Pipeline orchestration, scheduler, config (with GPU presets) |
| `crates/nordocr-server/src/` | 3 | HTTP API, CLI, main |
| `training/` | 4 | Python training/export scripts |
| `bench/benches/` | 1 | Criterion benchmarks |

## Key Design Decisions to Preserve

1. **Only two CPU↔GPU transfers** — raw image bytes in, text strings out. Don't add intermediate readbacks.
2. **Slab allocator** — never call cudaMalloc/cudaFree during inference (they synchronize the device).
3. **Width-sorted batching for recognition** — lines sorted by width before grouping into batches to minimize padding waste.
4. **CUDA Graphs** — capture the full pipeline once, replay with near-zero launch overhead.
5. **Multi-arch fat binaries** — CUDA kernels compiled for both sm_89 (A6000 Ada) and sm_120 (RTX 6000 PRO Blackwell). Set `NORDOCR_CUDA_ARCHS` to override.
6. **FP8 inference on Blackwell, FP16 on Ada** — TensorRT engines built with `--fp8` for sm_120, `--fp16` for sm_89. Separate engine files per architecture.
7. **Nordic charset as distinct letters** — å/ä/ö/ø/æ are NOT accented variants of a/o/e. The charset, training data, and evaluation metrics all treat them as separate letters.
