# SVTRv2 Retraining Guide: 768px → 1024px

## Why Retrain

Benchmark on 18-page test doc (`06313923_001_20260209T120358.tif`, 875 regions):

| Metric | SVTRv2 768px | Tesseract LSTM | Gap |
|--------|-------------|----------------|-----|
| Exact match | 659/875 (75.3%) | 691/875 (79.0%) | -32 |
| Space-insensitive | 748/875 (85.5%) | 792/875 (90.5%) | -44 |
| CER | 5.46% | 5.38% | +0.08pp |

SVTRv2 error breakdown (216 failures):

| Category | Count | % | Actionable? |
|----------|-------|---|-------------|
| Space-only (VLM artifact) | 89 | 41% | No — VLM strips grouping spaces |
| Minor char errors | 48 | 22% | Partially — more training data |
| Major errors | 38 | 18% | Partially |
| **Right truncation** | **27** | **12%** | **Yes — wider training** |
| **Leading garbage** | **13** | **6%** | **Yes — detection fix** |
| Trailing garbage | 1 | 0% | No |

**Right truncation is the #1 addressable issue.** The 768px model cuts text short on
long lines: "Summa kortfristiga sk" (should be "skulder"), "Kassaflöde från den
löpande verksa" (should be "verksamheten"). Retraining at 1024px would fix most of
these 27 cases, bringing SVTRv2 to ~79-80% exact match.

## Current Model Architecture

- **Model**: SVTRv2 (CTC-based, fully convolutional, variable-width input)
- **Framework**: PyTorch 2.10.0+cu130
- **ONNX opset**: 17
- **Input**: `[batch, 3, 32, width]` FP32 — batch and width are dynamic
- **Output**: `[batch, (width-1)//4 + 1, 126]` FP32 — CTC stride 4
- **Classes**: 126 (CTC blank at index 0, 125 chars at indices 1-125)
- **Model size**: ~79MB weights (ONNX data), ~43MB TRT engine (FP16)
- **Normalization**: `(pixel / 127.5) - 1.0` → range [-1, 1]. **NOT ImageNet stats!**

## Character Set (CRITICAL — must match exactly)

The CTC charset has 126 classes. Index 0 = CTC blank, indices 1-125 = characters.
Order must match `crates/nordocr-recognize/src/charset.rs` `CTC_CHARSET` exactly:

```
Index  1-10:  0123456789
Index 11-36:  ABCDEFGHIJKLMNOPQRSTUVWXYZ
Index 37-62:  abcdefghijklmnopqrstuvwxyz
Index 63-70:  ÅÄÖØÆÐÞÜ
Index 71-78:  åäöøæðþü
Index 79-111: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~  (space is index 79)
Index 112-125: §°€£«»–—''""±×
```

Generate `nordic_dict.txt` with one character per line in this exact order (125 lines).
The training code indexes into this file: `dict[i]` maps to CTC output index `i+1`.

## What to Change for 1024px

### Training Data Preparation

```python
class AspectPreservingResize:
    """Resize preserving aspect ratio, pad/crop to target width."""
    def __init__(self, target_h=32, max_w=1024):  # WAS: max_w=768
        self.target_h = target_h
        self.max_w = max_w

    def __call__(self, img):
        h, w = img.shape[:2]
        scale = self.target_h / h
        new_w = round(w * scale)
        new_w = (new_w // 4) * 4  # align to CTC stride
        new_w = max(4, min(new_w, self.max_w))

        img = cv2.resize(img, (new_w, self.target_h), interpolation=cv2.INTER_LINEAR)

        # Pad to max_w with WHITE (255). NOT gray!
        # Gray padding (128) causes model to corrupt right edge features.
        padded = np.ones((self.target_h, self.max_w, 3), dtype=np.uint8) * 255
        padded[:, :new_w] = img
        return padded
```

**Key points:**
- Change `max_w` from 768 → **1024**
- Padding color: **white (255, 255, 255)** → becomes 1.0 after normalization
- Normalize: `img.float() / 127.5 - 1.0` → range [-1, 1]
- Width alignment to 4 (CTC stride)

### Training Hyperparameters

| Parameter | 768px | 1024px | Notes |
|-----------|-------|--------|-------|
| `max_width` | 768 | **1024** | Training image padding width |
| `batch_size` | ~1024/GPU (A6000) | **~768/GPU** | 1024px uses ~33% more memory |
| | ~2048/GPU (Blackwell) | **~1536/GPU** | Scale proportionally |
| `max_seq_len` | 192 (768/4) | **256 (1024/4)** | CTC output length |
| `input_height` | 32 | 32 | Unchanged |
| `num_classes` | 126 | 126 | Unchanged |
| `epochs` | 20 | 20 | May need adjustment |
| `lr` | 7e-4 | 7e-4 | Unchanged |
| Normalization | (v/127.5)-1 | (v/127.5)-1 | Unchanged |

### ONNX Export

```bash
# Export from PyTorch checkpoint to ONNX
python export_onnx.py \
    --checkpoint best_model.pth \
    --output recognize_svtrv2_1024.onnx \
    --max-width 1024

# Simplify (MANDATORY for PARSeq, optional but recommended for SVTRv2)
python -m onnxsim recognize_svtrv2_1024.onnx recognize_svtrv2_1024_sim.onnx
```

ONNX export must use dynamic axes:
```python
dynamic_axes = {
    "input":  {0: "batch", 3: "width"},
    "output": {0: "batch", 1: "seq_len"},
}
dummy_input = torch.randn(1, 3, 32, 1024)
torch.onnx.export(model, dummy_input, "recognize_svtrv2_1024.onnx",
                  opset_version=17,
                  input_names=["input"],
                  output_names=["output"],
                  dynamic_axes=dynamic_axes)
```

### TensorRT Engine Build

Build engines for each target GPU:

```bash
# ============================================
# sm_86 (A6000 Ampere) — dev machine
# ============================================
trtexec \
    --onnx=recognize_svtrv2_1024.onnx \
    --saveEngine=recognize_svtrv2_1024_sm86.engine \
    --fp16 \
    --minShapes=input:1x3x32x32 \
    --optShapes=input:32x3x32x768 \
    --maxShapes=input:64x3x32x1792

# ============================================
# sm_120 (RTX 6000 PRO Blackwell) — production
# ============================================
trtexec \
    --onnx=recognize_svtrv2_1024.onnx \
    --saveEngine=recognize_svtrv2_1024_sm120.engine \
    --fp16 \
    --minShapes=input:1x3x32x32 \
    --optShapes=input:64x3x32x768 \
    --maxShapes=input:128x3x32x1792
```

**Shape notes:**
- `minShapes`: 1x3x32x32 — minimum for initialization
- `optShapes`: batch 32-64, width 768 — most common inference size
- `maxShapes`: width **1792** — must cover the widest lines in production
- The opt width (768) should match the most common inference width for best perf

### Rust Config Updates (after engine is ready)

In `crates/nordocr-pipeline/src/config.rs`, update presets:

```rust
// Change engine path
recognize_engine_path: "models/recognize_svtrv2_1024_sm86.engine",

// recognize_max_seq_len stays 448 (covers max_input_width 1792 / stride 4)
// recognize_max_input_width stays 1792
// No other Rust changes needed — engine auto-detects dims
```

## File Locations

| File | Path |
|------|------|
| Current 768px ONNX | `models/svtrv2-768/recognize_svtrv2.onnx` |
| Current 768px engine (sm86) | `models/recognize_svtrv2_768_sm86.engine` |
| New 1024px ONNX (after training) | `models/svtrv2-1024/recognize_svtrv2.onnx` |
| New 1024px engine (sm86) | `models/recognize_svtrv2_1024_sm86.engine` |
| New 1024px engine (sm120) | `models/recognize_svtrv2_1024_sm120.engine` |
| CTC charset (Rust, reference) | `crates/nordocr-recognize/src/charset.rs` |
| CTC charset (training dict) | `training/recognize/nordic_dict.txt` (generate) |
| Inference batch code | `crates/nordocr-recognize/src/batch.rs` |
| Inference engine code | `crates/nordocr-recognize/src/engine.rs` |

## Validation After Training

### Quick smoke test

```bash
# Setup PATH for TRT + CUDA
TRT_LIBS="C:/Users/Jens Nylander/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/tensorrt_libs"
CUDA_BIN="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64"
export PATH="$TRT_LIBS:$CUDA_BIN:$PATH"

# Build (Windows — use forward slashes, no trailing slash on TENSORRT_LIB_DIR!)
TENSORRT_LIB_DIR="C:/Users/Jens Nylander/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/tensorrt_libs" \
  cargo build --release

# Process single page
nordocr process "C:\Temp\06313923_pages\page02.png" \
    --format json --log-level error \
    --config /path/to/config_1024.json
```

### Full 3-way comparison

```bash
# Convert TIFF to PNGs (already done, at C:\Temp\06313923_pages\)
# Run SVTRv2 1024px on all pages, Tesseract on all pages
python3 scripts/run_benchmark.py   # saves svtrv2_result.json + tess_result.json

# 3-way comparison with VLM
python3 scripts/compare_3way.py "C:\Temp\06313923_001_20260209T120358.tif" \
    --svtrv2 C:\Temp\svtrv2_result_1024.json \
    --tess C:\Temp\tess_result.json \
    --output C:\Temp\3way_1024.json --workers 8
```

### Expected improvements

| Metric | 768px (current) | 1024px (expected) |
|--------|-----------------|-------------------|
| Right truncation | 27 cases | ~5-8 cases |
| Exact match | 75.3% | ~79-81% |
| Space-insensitive | 85.5% | ~88-90% |
| CER | 5.46% | ~3.5-4.5% |

## Gotchas and Hard-Won Lessons

1. **Normalization MUST be `(v/127.5)-1.0`**, NOT ImageNet `(v-mean)/std`. If wrong,
   the model outputs garbage.

2. **White padding (255→1.0), NOT gray (128→0.0).** Gray padding corrupts the right
   edge of the feature map and causes truncation. This was a bug we fixed in `batch.rs`.

3. **Charset order must match exactly.** Even one character off causes systematic
   misclassification. Generate `nordic_dict.txt` from the Rust `CTC_CHARSET` array.

4. **onnxsim is mandatory for PARSeq** (NaN without it). For SVTRv2 it's optional but
   recommended for cleaner TRT conversion.

5. **Per-item CTC decode is critical.** When batching, each item in the batch has a
   different actual width. The CTC decode must use `item_width / 4` as seq_len, not
   `batch_width / 4`. Using batch-wide seq_len causes 91 trailing garbage regions.
   This is already handled in `decode.rs:decode_cpu_per_item()`.

6. **TENSORRT_LIB_DIR must NOT have a trailing slash** on Windows, or the build.rs
   `PathBuf::from().join().exists()` check fails silently.

7. **TRT engine is GPU-arch specific.** Build separately for sm_86 (A6000) and
   sm_120 (Blackwell). The ONNX model is portable.

8. **Width alignment to 4.** Both training and inference must align width to 4 pixels
   (CTC stride). Misalignment causes off-by-one in output seq_len.
