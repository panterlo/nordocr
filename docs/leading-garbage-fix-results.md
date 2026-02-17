# Leading Garbage Fix — Results (2026-02-17)

## Problem

3-way comparison (SVTRv2 vs Tesseract vs VLM) on 18-page test document revealed
**52 regions (5.9%) with leading garbage** in SVTRv2 output. Root cause: horizontal
dilation (kernel_w=20, iterations=2) bridges fragments from adjacent text lines
into the same CCL component.

| Category | Count | Example | Root cause |
|----------|-------|---------|------------|
| Single leading space | 23 (44%) | `" Resultat..."` | Model reads bbox whitespace |
| Char+space prefix | 21 (40%) | `"da med den..."` | Dilation merges adjacent line fragment |
| Single char prefix | 8 (16%) | `"eRäntekostnader"` | Single char from adjacent line, no gap |

## Fix Applied

### Part 1: Vertical center-of-mass check in detection (`contour.rs`)

Added independent edge check in `trim_edge_fragments()`: for each edge word-cluster,
compute vertical center of mass vs the main body using the pre-dilation binary image.
If vertical centers differ by more than `max(4.0, bbox_height / 8.0)` pixels, the
fragment is from a different text line and gets trimmed.

This check is independent of the existing outlier-gap check. Both run, and either
can trigger a trim.

### Part 2: Whitespace stripping in recognition (`decode.rs`, `batch.rs`)

Added `DecodedText::strip_whitespace()` that strips leading/trailing spaces and
adjusts `char_confidences`, `char_positions`, and recomputes `confidence`. Called
after `trim_trailing_by_position()` in the recognition batch pipeline.

## Files Changed

- `crates/nordocr-detect/src/contour.rs` — added `vertical_center_of_mass()` helper,
  restructured `trim_edge_fragments()` with two-check approach
- `crates/nordocr-recognize/src/decode.rs` — added `DecodedText::strip_whitespace()`
- `crates/nordocr-recognize/src/batch.rs` — call `strip_whitespace()` after decode

## Verification Commands

```bash
# Setup
TRT_LIBS="/c/Users/Jens Nylander/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/tensorrt_libs/"
CUDA_BIN="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/x64/"
export PATH="$TRT_LIBS:$CUDA_BIN:$PATH"

# Build
TENSORRT_LIB_DIR="$TRT_LIBS" cargo build --release --features tesseract

# Convert CCITT Fax4 TIFF to per-page PNGs (Rust tiff crate doesn't support Fax4)
python3 -c "
from PIL import Image; import os
img = Image.open(r'C:\Temp\06313923_001_20260209T120358.tif')
os.makedirs(r'C:\Temp\06313923_pages', exist_ok=True)
for i in range(18):
    img.seek(i)
    img.convert('L').save(os.path.join(r'C:\Temp\06313923_pages', f'page{i:02d}.png'))
"

# Run SVTRv2 on all pages (config points to 768px sm86 engine)
# Config: C:\Temp\svtrv2_config.json with recognize_engine_path=models/recognize_svtrv2_768_sm86.engine
for i in $(seq -w 0 17); do
  nordocr process "C:\Temp\06313923_pages\page${i}.png" \
    --format json --config "C:\Temp\svtrv2_config.json" --log-level error
done
# -> merged into C:\Temp\svtrv2_result_v2.json

# Run Tesseract on all pages (same detection bboxes, different recognition)
for i in $(seq -w 0 17); do
  nordocr process "C:\Temp\06313923_pages\page${i}.png" \
    --format json --config "C:\Temp\svtrv2_config.json" \
    --recognize tesseract --tesseract-dll "C:\Dev\Ormeo\Ormeo.Tesseract\x64\tesseract55.dll" \
    --tessdata "C:\Tessdata_best" --tess-lang swe_ormeo_v3 --log-level error
done
# -> merged into C:\Temp\tess_result_v2.json

# 3-way comparison with VLM (Qwen3 VL 30B at http://10.12.2.31:8001/v1)
python3 scripts/compare_3way.py "C:\Temp\06313923_001_20260209T120358.tif" \
  --svtrv2 "C:\Temp\svtrv2_result_v2.json" \
  --tess "C:\Temp\tess_result_v2.json" \
  --output "C:\Temp\3way_v2.json" --workers 8
```

## Results

### Leading garbage after fix

| Metric | Before | After |
|--------|--------|-------|
| Leading spaces | 23 | **0** |
| Char+space prefix | 21 | **~5** |
| Single char prefix | 8 | **~0** |
| **Total leading garbage** | **52** | **~5** |

~90% reduction in leading garbage.

### 3-Way Comparison: SVTRv2 vs Tesseract vs VLM

875 regions across 18 pages. VLM used as ground truth.

| Metric | SVTRv2 | Tesseract |
|--------|--------|-----------|
| Exact match | 640/875 (**73.1%**) | 691/875 (**79.0%**) |
| Space-insensitive | 726/875 (**83.0%**) | 792/875 (**90.5%**) |
| Avg CER | **5.75%** | **4.94%** |
| Wins (other wrong) | 25 | 76 |
| Both wrong | 159 | 159 |

### Per-page breakdown

| Page | Rgns | SVT exact% | SVT CER | TSS exact% | TSS CER |
|------|------|-----------|---------|-----------|---------|
| 0 | 13 | 61.5% | 17.87% | 61.5% | 16.95% |
| 1 | 20 | 80.0% | 6.96% | 80.0% | 6.21% |
| 2 | 74 | 75.7% | 5.48% | 78.4% | 3.54% |
| 3 | 16 | 75.0% | 7.86% | 75.0% | 6.61% |
| 4 | 66 | 84.8% | 3.40% | 81.8% | 2.98% |
| 5 | 63 | 68.3% | 7.72% | 65.1% | 7.44% |
| 6 | 55 | 72.7% | 7.21% | 78.2% | 6.01% |
| 7 | 70 | 74.3% | 4.51% | 77.1% | 4.39% |
| 8 | 32 | 68.8% | 3.10% | 87.5% | 0.22% |
| 9 | 46 | 56.5% | 3.06% | 82.6% | 0.94% |
| 10 | 41 | 80.5% | 1.34% | 87.8% | 0.80% |
| 11 | 63 | 87.3% | 4.17% | 85.7% | 6.22% |
| 12 | 66 | 81.8% | 4.62% | 83.3% | 3.84% |
| 13 | 85 | 88.2% | 2.65% | 92.9% | 1.76% |
| 14 | 50 | 66.0% | 7.73% | 66.0% | 7.92% |
| 15 | 14 | 42.9% | 23.59% | 57.1% | 22.24% |
| 16 | 53 | 62.3% | 2.51% | 81.1% | 0.95% |
| 17 | 48 | 41.7% | 16.05% | 64.6% | 13.98% |

### Remaining SVTRv2 weaknesses

1. **Truncated text** (biggest issue): many lines cut short at the right edge.
   Examples: "Sala ko" (should be "Sala kommun."), "kortfristiga sk" (should be
   "kortfristiga skulder"). Likely the 768px model hitting its effective width limit
   on wide text regions.

2. **Remaining leading garbage** (~5 cases): fragments where the garbage char sits
   at a similar vertical position as the main text, so the center-of-mass threshold
   isn't triggered. Examples: "p frtfarande...", "pea marginalerna...", "pet Kassaflöde..."

3. **Space handling**: model collapses thin spaces ("12,13" vs "12, 13"), contributes
   to 86 space-only mismatches.

### SVTRv2 strengths vs Tesseract

- Better at Swedish diacriticals: Tesseract confuses "h"→"b" frequently
  ("handelsvaror"→"bandelsvaror", "hänförlig"→"bänförlig")
- No spurious space insertions ("Justeringar" vs "J usteringar")
- Better on pages 4, 5, 11, 14 (SVTRv2 higher exact%)
