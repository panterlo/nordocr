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

**All 10 crates compile (`cargo check` clean on sm_120).** First compilation pass completed on the Blackwell production server (2x RTX 6000 PRO Blackwell, CUDA 13.0, Driver 580.95.05). All CUDA kernels compile to PTX + fatbin. Warnings only — no errors.

### Compilation verified on

| Machine | GPUs | CUDA | Driver | Rust | Status |
|---|---|---|---|---|---|
| Blackwell prod | 2x RTX 6000 PRO Blackwell | 13.0 | 580.95.05 | 1.93.1 | `cargo check` clean |
| Ada dev | A6000 Ada | TBD | TBD | TBD | Not yet tested |

### What exists and is substantive

- **nordocr-core** — fully implemented types (`BBox`, `TextLine`, `PageResult`, `Polygon`, `FileInput`, etc.), error types, `PipelineStage` trait
- **nordocr-gpu** — `GpuContext`, `GpuMemoryPool` (slab allocator avoiding cudaMalloc during inference), `GpuBuffer<T>` (typed GPU memory wrapper), `StreamPool`. **Uses cudarc 0.19 API** (`CudaContext` + `CudaStream`).
- **nordocr-trt-sys** — bindgen build.rs (gated behind `generated` feature flag) + stub types that compile without TensorRT headers
- **nordocr-trt** — safe wrappers: `TrtRuntime`, `TrtEngine`, `TrtExecutionContext`, `TrtEngineBuilder` (FP8/FP4/sparsity config), `CudaGraph` capture/replay
- **nordocr-preprocess** — 4 CUDA kernels (.cu files with real algorithms) + Rust wrappers + `PreprocessPipeline` orchestrating denoise→deskew→binarize→morphology. **Multi-arch build**: compiles PTX for target arch, generates stubs for others. Runtime GPU detection via `GpuArch::detect()`.
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

### Fixes applied during first compilation (sm_120)

These are already done — listed for reference so you don't re-investigate:

| Issue | Fix applied |
|---|---|
| cudarc 0.12 doesn't support CUDA 13.0 | Upgraded to cudarc 0.19 with `cuda-13000` feature |
| `CudaDevice` removed in cudarc 0.19 | `CudaDevice` → `CudaContext`, allocations via `CudaStream` not device |
| `DevicePtr::device_ptr()` changed signature | Now takes `&CudaStream`, returns `(ptr, SyncOnDrop)` — scope the borrow guard |
| `device.htod_sync_copy()` removed | → `stream.clone_htod()` |
| `device.alloc_zeros()` removed | → `stream.alloc_zeros()` |
| `GpuBuffer::from_cuda_slice` needs stream | Added `stream` param, caches raw pointer at creation time |
| `nvidia-dali` workspace dep `optional = true` | Removed — `optional` not valid on workspace deps |
| `half::f16` missing `bytemuck::Pod` impl | Added `bytemuck` feature to `half` workspace dep |
| `axum::Multipart` not found | Added `multipart` feature to `axum` workspace dep |
| Unicode curly quotes in charset.rs | `'` `'` → `\u{2018}` `\u{2019}` escape sequences |
| Missing sm_89 PTX when building sm_120-only | build.rs now generates stubs for non-compiled architectures |

## Building on the Ada Machine (sm_89)

### 1. Prerequisites

```bash
# Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# CUDA must be on PATH — adjust path for your CUDA version
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Verify
nvcc --version          # need CUDA 12.x+
nvidia-smi              # confirm A6000 Ada visible
rustc --version         # need 1.75+
```

### 2. Build

```bash
cd nordocr

# Build for Ada only (faster — skips Blackwell PTX compilation)
NORDOCR_CUDA_ARCHS="sm_89" cargo check

# Build for both architectures (fat binary, default)
cargo check

# Build for Blackwell only
NORDOCR_CUDA_ARCHS="sm_120" cargo check
```

If your CUDA version is 12.x, the cudarc `cuda-13000` feature may fail. Change it in `Cargo.toml`:
```toml
# For CUDA 12.6:
cudarc = { version = "0.19", features = ["driver", "nvrtc", "cuda-12060"] }
# For CUDA 12.8:
cudarc = { version = "0.19", features = ["driver", "nvrtc", "cuda-12080"] }
```

Available CUDA version features: `cuda-11040` through `cuda-13010`. Use `cuda-version-from-build-system` for auto-detection (may not work on all setups).

### 3. Remaining warnings

All remaining compiler output is **warnings only** (unused imports, unused fields, unused variables). These are expected — the placeholder code references types it doesn't yet use. Don't spend time fixing these until wiring up real implementations.

### 4. Wire up real CUDA calls

The highest-value work now that `cargo check` is clean:

1. **nordocr-gpu**: Make `GpuContext::new`, `GpuMemoryPool::alloc/free`, `StreamPool` work with real cudarc calls. The cudarc 0.19 migration is done — the types are correct, but the actual CUDA operations are placeholders. This unblocks everything.

2. **nordocr-preprocess**: Load compiled PTX via `cudarc`, launch kernels. The `.cu` files have real algorithms — the Rust wrappers just need real `launch!` calls. The `GpuArch` detection in `gpu_arch.rs` can now use `ctx.compute_capability()` (cudarc 0.19 API).

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

## cudarc 0.19 API Quick Reference

The codebase uses cudarc 0.19 (not 0.12). Key types:

```rust
// Device/context
let ctx: Arc<CudaContext> = CudaContext::new(ordinal)?;
let (major, minor) = ctx.compute_capability()?;

// Streams
let stream: Arc<CudaStream> = ctx.default_stream();
let stream2: Arc<CudaStream> = ctx.new_stream()?;

// Allocation (on stream, not device)
let data: CudaSlice<f32> = stream.alloc_zeros(1024)?;
let data: CudaSlice<u8> = stream.clone_htod(&host_vec)?;

// Device pointer (needs stream, returns borrow guard)
let raw_ptr = {
    let (ptr, _guard) = DevicePtr::device_ptr(&slice, &stream);
    ptr as u64
}; // _guard dropped here, releasing borrow

// Synchronization
stream.synchronize()?;
ctx.synchronize()?;
```

## Multi-GPU-Architecture Build

```
build.rs (nordocr-preprocess)
  │
  ├── For each arch in NORDOCR_CUDA_ARCHS (default: sm_89,sm_120):
  │     nvcc --ptx → {kernel}_{arch}.ptx
  │
  ├── For archs NOT in NORDOCR_CUDA_ARCHS:
  │     generates stub PTX ("// STUB: ...")
  │
  └── nvcc --fatbin (all requested archs)
      → {kernel}.fatbin  (driver picks best code at load time)

Runtime (gpu_arch.rs):
  GpuArch::detect() → queries CUDA device compute capability
    sm_89  → load *_sm89.ptx
    sm_120 → load *_sm120.ptx (fallback to sm_89 if unavailable)
```

Override: `NORDOCR_TARGET_ARCH=sm_89` or `NORDOCR_TARGET_ARCH=sm_120`

## Crate Dependency Graph

```
nordocr-core          (no deps — shared types/errors)
  ├── nordocr-gpu     (cudarc 0.19 — GPU context, memory, streams)
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

---

## Experimental / Cutting-Edge Features

These are integrated into the codebase but should be benchmarked before production use.

### Model Alternatives

The pipeline supports swappable model architectures via `PipelineConfig`:

| Stage | Default | Alternatives | Trade-off |
|---|---|---|---|
| **Detection** | DBNet++ | RTMDet | RTMDet may be faster (single-stage, CSPNeXt backbone) but less tested on documents |
| **Recognition** | PARSeq | MAERec, CLIP4STR | MAERec: better on degraded scans (MAE pre-training). CLIP4STR: best font generalization (CLIP backbone) but ~5-10x slower |

Set via config:
```json
{
  "detect_model": "RTMDet",
  "recognize_model": "MAERec"
}
```

Training scripts for all variants are in `training/detect/` and `training/recognize/`.
Export all model variants + build engines for both GPUs: `python training/export/export_onnx_all.py`

### NVIDIA DALI (GPU-Accelerated Decode)

Replaces nvJPEG + manual resize + normalize with a single fused GPU pipeline.

- Feature-gated: `cargo build --features nordocr-decode/dali`
- Eliminates intermediate GPU memory allocations between decode stages
- Built-in double-buffered prefetch for batch workloads
- Falls back gracefully to standard decode if DALI unavailable
- Enabled via `PipelineConfig { enable_dali: true, .. }`
- Code: `crates/nordocr-decode/src/dali.rs`

### cuDLA (Deep Learning Accelerator) Offload

Blackwell GPUs include dedicated DLA cores separate from the GPU SMs. Offloading detection to DLA frees GPU SMs for preprocessing kernels — true hardware parallelism.

- Configured via `DlaConfig` in `nordocr-trt` builder
- `PipelineConfig::rtx6000_pro_blackwell_dla()` preset enables it
- Not available on A6000 Ada (sm_89)
- Recommended: offload detection to DLA, keep recognition on GPU SMs
- Code: `DlaConfig` in `crates/nordocr-trt/src/builder.rs`

### FP4 (NF4) Quantization

~2x throughput over FP8 for weight-bound layers. Blackwell (sm_120) only.

- Configured via `TrtEngineBuilder::with_fp4()`
- `PipelineConfig { precision: InferencePrecision::FP4, .. }`
- `PipelineConfig::rtx6000_pro_blackwell_experimental()` preset enables FP4 + DLA + DALI
- **WARNING**: PARSeq-S has only ~20M params — limited redundancy to absorb quantization error. FP4 may degrade diacritical accuracy. Validate on the Nordic test set before deploying.
- Better suited for the detection backbone (larger model, more redundancy)

### Configuration Presets

```rust
// Development (A6000 Ada, conservative)
PipelineConfig::a6000_ada()

// Production (RTX 6000 PRO Blackwell, FP8)
PipelineConfig::rtx6000_pro_blackwell()

// Production + DLA offload
PipelineConfig::rtx6000_pro_blackwell_dla()

// Experimental maximum throughput (FP4 + DLA + DALI)
PipelineConfig::rtx6000_pro_blackwell_experimental()
```

### Benchmarking Approach

For each experimental feature, the recommended evaluation:

1. **Baseline**: DBNet++ + PARSeq + FP16 on A6000 Ada
2. **Production**: DBNet++ + PARSeq + FP8 on RTX 6000 PRO Blackwell
3. **+DLA**: Same as #2 but with detection offloaded to DLA
4. **+FP4**: Same as #3 but with FP4 recognition
5. **+DALI**: Same as #4 but with DALI decode
6. **Alt models**: Swap DBNet++ for RTMDet, PARSeq for MAERec/CLIP4STR

Metrics per variant:
- Throughput (pages/sec)
- Latency (ms/page)
- CER, WER
- Diacritical accuracy (å/ä/ö/ø/æ)
- GPU memory usage
