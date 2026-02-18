# nordocr Training Pipeline

## Scope

nordocr does **text detection + character recognition** only. It is NOT a layout analysis system.

| Concern | Who handles it | What it does |
|---|---|---|
| **Layout analysis** | Ormeo.DocVision (existing) | Reading order, columns, tables, headers, CCL, projection profiles |
| **Text detection** | nordocr (DBNet++/FAST) | Finds WHERE text regions are on a page → bounding boxes |
| **Text recognition** | nordocr (PARSeq/SVTRv2) | Reads WHAT characters are in each cropped text line |

nordocr replaces the Tesseract LSTM step in Ormeo.DocVision's pipeline. Layout analysis and reading order remain in Ormeo's existing GPU kernel pipeline.

```
Ormeo.DocVision pipeline (existing):
  Scan → GPU preprocess → Layout analysis → [Tesseract LSTM] → Qwen3-VL spot-check → text
                                                    │
                                    nordocr replaces this step
```

---

## Data Sources

### Real scanned documents (SeaweedFS)

- **Volume**: ~10 million multi-page TIFF CCITT Group 4 files
- **Storage**: SeaweedFS, accessible via HTTP API from training scripts
- **Format**: 1-bit binary (already binarized by scanner), typically 200-300 DPI
- **Languages**: Swedish, Norwegian, Danish, Finnish (mixed within collection)
- **Content**: Nordic business/government documents (invoices, contracts, official letters, forms)

These are the **production inference workload**, not the full training set. A carefully selected subset is used for training.

### Tesseract high-confidence auto-labels (bulk training data)

- **Source**: Run Tesseract on a subset of real TIFFs from SeaweedFS (100K-500K pages)
- **Filter**: Keep only text lines where Tesseract reports confidence ≥ 95
- **Volume**: Millions of text line crops (50+ lines/page × hundreds of thousands of pages)
- **Ground truth quality**: ~98-99% correct overall. On diacriticals specifically, possibly ~95-98% — Tesseract can be confidently wrong on å/ä/ö/ø/æ
- **Why this works**:
  - Real data, real artifacts, real fonts — zero domain gap
  - Massive volume — even after strict filtering, you get millions of samples
  - 1-2% label noise is tolerable for neural network training when the clean signal dominates
  - Much simpler to build than a synthetic rendering pipeline
- **Risk**: Tesseract's systematic diacritical errors may leak through even at high confidence. Mitigated by the verified cutouts (see below)

### Existing labeled cutouts (spot-check corpus — gold standard)

- **Volume**: ~40,000 cropped text images with 100% verified ground truth
- **Origin**: Text regions where Tesseract LSTM reported low confidence, sent to spot-check (Qwen3-VL), verified and corrected
- **Ground truth quality**: Gold standard — human/VLM-verified, 100% accurate
- **Why they're especially valuable**:
  - Biased toward Tesseract's failure cases — disproportionately rich in diacriticals (å, ä, ö, ø, æ) and degraded text
  - Real scanner artifacts, real fonts, real DPI — zero domain gap with production data
  - Already cropped to text regions — exactly the format recognition models consume
  - These are the **antidote** to Tesseract's diacritical errors in the auto-labeled data
- **Format**: Cropped text images + verified text labels (details TBD: file naming, directory structure, label format)

### Synthetic data (supplementary, optional)

- **Volume**: Generated on demand, only for gaps not covered by real data
- **Ground truth**: Perfect by construction
- **Role**: Small targeted supplement for:
  - Rare characters that appear infrequently in the real corpus (þ, ð, €, «, »)
  - Forced diacritical minimal pairs ("hår"/"har", "för"/"for") if underrepresented in auto-labels
  - Edge case fonts or degradation patterns not in the real data
- **NOT the primary data source** — the Tesseract-bootstrapped real data provides better domain coverage

---

## The Ground Truth Strategy

We have 10M+ scanned pages, **40K already-labeled text cutouts**, and Tesseract already deployed. Rather than building a complex synthetic rendering pipeline, we bootstrap training data from what we have:

### Target volumes

| Dataset | Target samples | Role |
|---|---|---|
| Tesseract auto-labeled (confidence ≥ 95) | **200K** | Bulk real domain data |
| Verified spot-check cutouts (40K × 4 oversample) | **160K effective** | Gold standard, corrective signal on diacriticals |
| Synthetic (optional, rare chars only) | 10K-50K | Gap fill for þ, ð, €, «, » |
| **Effective training set** | **~360K-410K** | Solid generalization tier |

Volume tiers for reference:

| Tier | Auto-labeled samples | What you get |
|---|---|---|
| Minimum viable | 50K | Charset adaptation, basic diacritical handling |
| Solid (recommended) | **200K** | Good generalization across fonts, layouts, scan quality |
| Ideal | 500K | Maximum robustness, rare character coverage |

**Diversity is more important than raw volume.** Pull from a wide spread of SeaweedFS FIDs to cover:
- Different fonts (serif, sans-serif, monospace)
- Different scan qualities (clean vs degraded)
- Different document types (annual reports, invoices, contracts, forms, letters)
- All Nordic diacriticals well-represented (å, ä, ö, ø, æ)
- Numbers, dates, currency formats, mixed alpha-numeric text

### Primary approach: Tesseract-bootstrapped real data + verified cutouts

```
10M multi-page TIFFs (SeaweedFS)
   │
   ▼
Run Tesseract on ~5K-10K pages (→ ~200K line crops at ≥95 confidence)
   │
   ├──► confidence ≥ 95 ──► 200K auto-labeled line crops
   │                         ~98-99% correct overall, ~95-98% on diacriticals
   │
   └──► confidence < 95 ──► already covered by 40K verified cutouts
                             (new low-conf crops can also be sent to VLM for more verified data)

Training mix:
   ┌──────────────────────────────────────────────────────────────────────┐
   │  40K verified cutouts (×4 oversampling = 160K)  — gold, hard cases  │
   │  200K Tesseract high-confidence auto-labels     — bulk, real domain  │
   │  Targeted synthetic (optional, 10K-50K)         — rare chars only    │
   └──────────────────────────────────────────────────────────────────────┘
   Total effective: ~360K-410K samples
```

**Why this works better than synthetic-first:**

| | Synthetic-first approach | Tesseract-bootstrapped approach |
|---|---|---|
| Domain gap | Must simulate CCITT artifacts | Zero — it IS real data |
| Build effort | Complex rendering pipeline | Just run Tesseract (already deployed) |
| Volume | Unlimited but artificial | Millions of real samples |
| Font/layout coverage | Must manually select fonts | Automatic — whatever the real docs have |
| Label quality | Perfect (synthetic) | ~98-99% (high-confidence filter) |
| Diacritical correction | Perfect (synthetic) | ~95-98% — mitigated by 40K verified cutouts |

**The diacritical risk and its mitigation:**

Tesseract can be confidently wrong on diacriticals — it may output "a" with 98% confidence when the actual character is "å". The confidence score reflects Tesseract's certainty, not actual correctness. This means some diacritical errors will leak through even at high confidence thresholds.

This is mitigated by the 40K verified cutouts:
- They're specifically the cases where Tesseract got diacriticals wrong, now corrected
- Oversampled 3-5× in training, they provide a strong corrective signal
- The model sees "å" in its correct form 40K × 3-5 = 120K-200K times from verified data
- Versus maybe 50K-100K incorrect "a→å" errors in the auto-labeled bulk data
- The correct signal dominates, especially combined with the pretrained model's existing character knowledge

**Additional optional filters for auto-labeled data:**
- **Dictionary/spell-check pass**: Flag words where Tesseract output is valid but a diacritical variant is more likely in context (e.g., "har" is valid Swedish, but "hår" might be more likely based on surrounding words)
- **Diacritical frequency check**: If Tesseract's output for a document has suspiciously few diacriticals compared to typical Nordic text, flag it

### Fallback: VLM labeling oracle

If the Tesseract-bootstrapped approach proves insufficient (e.g., diacritical accuracy doesn't reach targets), the VLM labeling approach remains available:

- Run Qwen3-VL 32B (or DeepSeek-OCR-2, if Nordic evaluation passes) on a curated subset
- Use as higher-quality labels for the cases where Tesseract is unreliable
- This is now a **fallback**, not the primary approach — saving significant compute cost

---

## Detection Training (DBNet++/FAST)

Detection is about finding WHERE text is on the page. Ground truth is bounding boxes/polygons around text regions.

### Why detection labels are easy for CCITT TIFFs

CCITT Group 4 TIFFs are already 1-bit binary. Text detection on binary images is largely a solved problem:

1. **Morphological approach** (fully automatic, no ML needed):
   - Dilate the binary image (connect nearby text into blocks)
   - Find connected components
   - Filter by aspect ratio and size (remove noise, keep text blocks)
   - Output: bounding boxes around text regions
   - This gives you detection ground truth with ~95%+ accuracy

2. **Tesseract's bounding boxes** (also automatic):
   - Tesseract's text DETECTION is quite good even when its RECOGNITION is wrong
   - Extract word/line-level bounding boxes from Tesseract's TSV output
   - These are reliable detection labels even with noisy recognition

3. **Ormeo.DocVision's layout kernels** (best option):
   - You already have GPU-accelerated CCL, projection profiles, and bbox extraction
   - Run the existing Ormeo preprocessing + layout pipeline on a TIFF subset
   - Extract the text region bounding boxes it produces
   - These are production-tested on your actual document types

### Detection training data recipe

| Source | Volume | Labels | Quality |
|---|---|---|---|
| Ormeo layout pipeline on real TIFFs | 10K-50K pages | Bounding boxes from CCL + projection profiles | High (production-tested) |
| Synthetic full-page layouts | 50K-100K pages | Perfect (rendered at known positions) | Perfect but may not match real layouts |
| Tesseract TSV bounding boxes | 50K pages | Word/line boxes from Tesseract | Good for detection, even if text is wrong |

### Detection training config

```yaml
model: DBNet++  # or FAST
backbone: ConvNeXt-T  # pretrained on ImageNet
input_size: [1024, 1024]
epochs: 100
batch_size: 16
optimizer: AdamW
lr: 1e-4
scheduler: cosine
loss: L_prob + 0.5 * L_thresh + 1.0 * L_binary

# Data
train_data:
  - source: ormeo_layout_boxes    # real TIFF pages + Ormeo-generated bboxes
    weight: 0.6
  - source: synthetic_pages        # rendered full-page layouts
    weight: 0.4

# Augmentation
augmentation:
  rotation: [-5, 5]
  scale: [0.8, 1.2]
  noise: gaussian  # even though input is binary, add noise before binarization
```

---

## Recognition Training (PARSeq/SVTRv2)

Recognition is about reading WHAT characters are in a cropped text line image.

### Training pipeline overview

```
SeaweedFS TIFFs ──► Tesseract (confidence ≥ 95) ──► millions of auto-labeled line crops
                                                              │
40K verified spot-check cutouts (gold standard) ──────────────┤
                                                              │
Optional: targeted synthetic for rare chars ──────────────────┤
                                                              ▼
                                                    Mixed training set
                                                              │
                                                              ▼
                                              Fine-tune pretrained PARSeq/SVTRv2
                                                              │
                                                              ▼
                                                    Evaluate diacritical accuracy
                                                              │
                                            ┌─────────────────┴──────────────────┐
                                            │                                    │
                                    Meets targets?                       Below targets?
                                            │                                    │
                                    Export to ONNX/TRT               Expand with VLM labels
                                                                     or more verified cutouts
```

### Phase 1: Tesseract-bootstrapped training (weeks 1-2)

Fine-tune pretrained PARSeq/SVTRv2 on real data from the start — no synthetic phase needed.

**Step 1: Generate auto-labeled data from real TIFFs**

```
SeaweedFS (~5K-10K diverse pages → ~200K line crops at ≥95 confidence)
   │
   ▼
Run Tesseract LSTM with TSV output (word-level bboxes + text + confidence)
   │
   ├──► For each text line:
   │     - Crop the line region from the TIFF page
   │     - Record Tesseract's text output and confidence
   │     - Keep only lines where ALL words have confidence ≥ 95
   │
   ▼
Target: 200K auto-labeled (line_image, text_label) pairs
```

**Step 2: Combine with verified cutouts and train**

```yaml
# PARSeq (fixed-width 384px, NAR decoder):
model: PARSeq-S
charset: nordic_125  # 125 chars + EOS/BOS/PAD = 128 tokens
input_height: 32
max_input_width: 384   # fixed — images resized/padded to this width
max_seq_len: 64
epochs: 20
batch_size: 2048       # PARSeq-S is small, fits large batches
optimizer: AdamW, lr: 7e-4, scheduler: one_cycle

# SVTRv2 (variable-width up to 1792px, CTC decoder):
model: SVTRv2-base + RCTC
charset: nordic_125  # 125 chars + blank = 126 output classes
input_height: 32
max_input_width: 1792  # variable — no compression on wide lines
max_seq_len: 448       # = 1792 / 4 (stride 4)
max_label_len: 100     # increased from PARSeq's 25 for full-width lines
epochs: 20
batch_size: 384        # variable-width images, bigger per sample
optimizer: AdamW, lr: 2e-4, scheduler: one_cycle

# Both models share the same training data mix:
train_data:
  - source: tesseract_high_conf     # 200K auto-labeled from real TIFFs
    filter: confidence >= 95
    weight: 0.5                      # bulk real data
  - source: spot_check_cutouts      # 40K verified cutouts (gold standard)
    oversample: 4x                   # 40K × 4 = 160K effective samples
    weight: 0.4                      # strong corrective signal on diacriticals
  - source: synthetic_targeted       # optional: rare chars only (þ, ð, €, «, »)
    volume: 10_000-50_000            # small — only for charset gaps
    weight: 0.1
```

**Why oversample the 40K cutouts 4×?**

With 200K auto-labeled samples, the 40K gold-standard cutouts would be underrepresented without oversampling. At 4× oversampling:
- 40K × 4 = 160K verified samples in the training mix
- If auto-labeled data has ~1-2% diacritical errors = ~2K-4K wrong labels in 200K
- The 160K correct labels massively outnumber the wrong ones on diacriticals
- The model learns to trust the verified patterns over Tesseract's errors

### Phase 2: Evaluate and iterate (week 3)

Run the trained model on held-out test set. Check diacritical accuracy specifically.

**If diacritical accuracy ≥ 99%:** Done — export to ONNX/TensorRT.

**If diacritical accuracy < 99%:** Options to improve:
1. **Increase confidence threshold** (e.g., ≥ 98) — reduces noise but also reduces volume
2. **Increase cutout oversampling** (e.g., 10×) — stronger corrective signal
3. **Add VLM-labeled data** — run Qwen3-VL or DeepSeek-OCR-2 on a subset of pages where Tesseract reported moderate confidence (70-95) to get more verified diacritical samples
4. **Add targeted synthetic data** — generate synthetic samples specifically for underperforming diacritical pairs

### Phase 3: Active learning (ongoing, post-deployment)

Deploy the model and continuously improve:

```
Production inference on new documents
   │
   ├──► High confidence (>0.95) → accept, optionally add to training set
   │
   ├──► Medium confidence (0.7-0.95) → accept but flag for review queue
   │
   └──► Low confidence (<0.7), especially on diacriticals → human/VLM review
              │
              ▼
         Corrected labels → add to training set
              │
              ▼
         Periodic retraining (monthly or as correction volume grows)
```

---

## Synthetic Data Generation (Supplementary Role)

Synthetic data is **no longer the primary training data source**. The Tesseract-bootstrapped real data provides better domain coverage. Synthetic generation is used only for:

1. **Rare characters** not frequently seen in real documents (þ, ð, €, «, », §, °)
2. **Forced diacritical pairs** if evaluation shows specific confusion patterns need more training signal
3. **Fallback** if the Tesseract-bootstrapped approach doesn't reach accuracy targets

If needed, the synthetic pipeline generates text line images using Nordic text corpora rendered with fonts matching the real documents, with CCITT-like degradation (1-bit binarization, speckle noise, thin stroke dropout). See `training/data/generate_nordic_synthetic.py` for the stub implementation.

---

## Data Preparation Pipeline (C# — Ormeo.Document)

Training data preparation runs in the **existing C# Ormeo.Document pipeline**, not in Python or Rust. The C# codebase already has all required integrations:

| Integration | Existing C# component |
|---|---|
| SeaweedFS file download | `SeaweedFSClient` (Ormeo.Library) |
| Multi-page TIFF CCITT G4 splitting | `TiffSplitterStep` (ImageMagick/Magick.NET) |
| Tesseract OCR with confidence scores | `TextRegionOcrStep` (Ormeo.Tesseract) |
| Text region detection (bboxes) | OpenCvSharp in `TextRegionOcrStep` |
| MSSQL database access | Entity Framework Core (`OrmeoContext`) |
| 40K spot-check cutouts | `TesseractLineImage` table (varbinary + validated text) |

### Architecture

```
C# (Ormeo.Document)              Python (training/)              Rust (nordocr)
─────────────────────             ──────────────────              ──────────────
DATA PREPARATION                  MODEL TRAINING                  INFERENCE

SeaweedFS → download TIFFs        Load prepared dataset           Load TensorRT engines
TiffSplitterStep → split pages    Fine-tune PARSeq/SVTRv2         Run production OCR
TextRegionOcrStep → Tesseract     Evaluate diacritical accuracy   <50ms/page target
  + confidence scores             Export ONNX → TensorRT
Crop text lines from bboxes
Filter confidence ≥ 95
Export 40K cutouts from MSSQL

Output: training_data/            Output: models/*.onnx           Output: text
  ├── auto_labeled/               → trtexec → *.engine
  │   ├── images/
  │   └── labels.tsv
  └── spot_check/
      ├── images/
      └── labels.tsv
```

### New pipeline step: TrainingDataExportStep

Add a new step to `Ormeo.Document/Pipeline/Steps/` that:

1. **Downloads N TIFFs from SeaweedFS** (configurable batch, e.g., 100K-500K pages)
2. **Splits multi-page TIFFs** (reuse existing `TiffSplitterStep`)
3. **Runs Tesseract** on each page with TSV output (word-level bboxes + text + confidence)
4. **Filters by confidence**: keep text lines where all words have confidence ≥ 95
5. **Crops text line images** from the page using Tesseract's bounding boxes
6. **Saves to disk**: image file + label in a format Python can read

```
Output format:
training_data/auto_labeled/
├── images/
│   ├── 000001.png    # cropped text line (height normalized to 32px)
│   ├── 000002.png
│   └── ...
└── labels.tsv        # tab-separated: filename \t text \t confidence \t source_fid
```

### Spot-check cutout export

A separate command/step that:

1. **Queries MSSQL** for the 40K `TesseractLineImage` records
2. **Extracts varbinary** image data + validated text labels
3. **Saves to disk** in the same format as auto-labeled data

```
Output format:
training_data/spot_check/
├── images/
│   ├── sc_000001.png
│   ├── sc_000002.png
│   └── ...
└── labels.tsv        # tab-separated: filename \t verified_text
```

### Detection label export

For detection training, export full pages with bounding box annotations:

1. **Run Tesseract** with TSV output → extract line-level bounding boxes
2. **Or use Ormeo's OpenCV pipeline** → text region contours
3. **Save in COCO format** (JSON with image paths + polygon annotations)

```
Output format:
training_data/detection/
├── images/           # full page images (resized to 1024x1024)
└── annotations.json  # COCO-format bbox/polygon annotations
```

### Why C# and not Python/Rust for data prep

| Concern | C# (Ormeo.Document) | Python | Rust (nordocr) |
|---|---|---|---|
| SeaweedFS access | Already built | Rewrite HTTP client | Rewrite HTTP client |
| TIFF CCITT G4 handling | ImageMagick (tested) | Pillow (may struggle with CCITT) | image crate (limited TIFF) |
| Tesseract integration | Tested, with confidence | pytesseract (thinner wrapper) | tesseract-sys (would need FFI) |
| MSSQL for cutouts | Entity Framework | pyodbc (extra dep) | tiberius (extra dep) |
| Text region detection | OpenCvSharp (tested) | OpenCV (would work) | Not available |
| Build effort | Add one pipeline step | Rewrite all integrations | Rewrite all integrations |
| Run frequency | Once (batch job) | Once (batch job) | Once (batch job) |

The data prep runs once (or periodically for retraining). It doesn't need GPU speed — it's I/O bound (SeaweedFS download + Tesseract CPU). The existing C# pipeline has years of battle-tested integrations. Don't rebuild what already works.

---

## Evaluation Metrics

### Standard OCR metrics

| Metric | What it measures | Target |
|---|---|---|
| CER (Character Error Rate) | % of characters wrong | < 1% (beat Tesseract's ~3-5% on Nordic) |
| WER (Word Error Rate) | % of words with any error | < 3% |
| Precision/Recall @ IoU 0.5 | Detection accuracy | > 95% F1 |

### Nordic-specific metrics (the whole point)

| Metric | What it measures | Target |
|---|---|---|
| Diacritical accuracy | Correct å/ä/ö/ø/æ vs base letter confusion | > 99% |
| Confusion matrix: a↔å↔ä | How often a, å, ä are confused with each other | < 0.5% confusion rate |
| Confusion matrix: o↔ö↔ø | How often o, ö, ø are confused with each other | < 0.5% confusion rate |
| Nordic CER | CER measured only on words containing diacriticals | < 1% |

### Benchmark against baseline

Every model version must be benchmarked against:
1. **Tesseract LSTM** (current production) — the bar to beat
2. **Tesseract + Qwen3-VL spot-check** (current full pipeline) — quality ceiling to approach
3. **Previous nordocr version** — regression check

### Test set

A held-out set of ~5K pages from real TIFFs, labeled by Qwen3-VL + human verification on a subset. This test set is NEVER used for training.

---

## Directory Structure

### Data preparation (C# — in Ormeo.Document)

```
C:\Dev\Ormeo\Ormeo.Document\Pipeline\Steps\
└── TrainingDataExportStep.cs          # NEW: exports training data from SeaweedFS + Tesseract (TODO)

C:\Dev\Ormeo\Ormeo.Document\Tools\Tesseract\
└── TesseractHandler.cs                # EXISTING: already has line image extraction methods
```

### Training output (shared disk/storage)

```
training_data/                          # Output from C# pipeline, consumed by Python
├── auto_labeled/                       # Tesseract high-confidence line crops
│   ├── images/                         # PNG files, height-normalized to 32px
│   └── labels.tsv                      # filename \t text \t confidence \t source_fid
├── spot_check/                         # 40K verified cutouts from MSSQL
│   ├── images/                         # PNG files from TesseractLineImage varbinary
│   └── labels.tsv                      # filename \t verified_text
└── detection/                          # Full pages with bbox annotations
    ├── images/                         # Full page PNGs (resized to 1024x1024)
    └── annotations.json                # COCO-format bounding box annotations
```

### Model training (Python — in nordocr/training/)

```
nordocr/training/
├── TRAINING_PIPELINE.md               ← this document
├── requirements.txt                    # pip deps (torch, lmdb, onnx, etc.)
├── data/
│   └── generate_nordic_synthetic.py   # Supplementary synthetic generation (optional)
├── detect/
│   ├── train.py                       # DBNet++ fine-tuning
│   └── train_fast.py                  # FAST fine-tuning
├── recognize/
│   ├── charset.py                     # NordicTokenizer (125 chars + 3 specials)
│   ├── dataset.py                     # TSV+images dataset with AspectPreservingResize
│   ├── train_parseq.py                # PARSeq fine-tuning (standalone, no external deps)
│   ├── train_svtrv2.py                # SVTRv2 fine-tuning (via OpenOCR)
│   └── nordic_dict.txt                # 125-char dictionary for CTC/OpenOCR
├── tools/
│   ├── split_dataset.py               # Split TSV+images → train/val/test
│   └── tsv_to_lmdb.py                 # Convert TSV+images → LMDB for OpenOCR
├── export/
│   ├── export_onnx.py                 # PARSeq ONNX export (fixed width, needs onnxsim)
│   ├── export_onnx_svtrv2.py          # SVTRv2 ONNX export (dynamic width, no onnxsim)
│   ├── export_onnx_all.py             # Bulk export + TRT build for all models/GPUs
│   └── build_trt_engine.py            # ONNX → TensorRT engine builder
└── eval/
    └── evaluate.py                    # Full Nordic eval: --model parseq|svtrv2
                                       #   CER, WER, diacritical accuracy, confusion matrix,
                                       #   per-char breakdown, length-bucketed accuracy,
                                       #   worst-50 predictions for manual review
```

---

## Execution Order

```
Step 1: Generate Tesseract auto-labeled data from real TIFFs  [C# — Ormeo.Document]
──────────────────────────────────────────────────────────────
Run TrainingDataExportStep in Ormeo.Document pipeline:
  - Download ~5K-10K diverse pages from SeaweedFS (varied document types, FIDs)
  - Split multi-page TIFFs (TiffSplitterStep)
  - Run Tesseract with TSV output (bboxes + text + confidence)
  - Filter: keep lines with all-word confidence ≥ 95
  - Crop text lines, normalize height to 32px
  - Target: 200K auto-labeled line crops
  → training_data/auto_labeled/  (200K line_image + label pairs)

Step 2: Export 40K spot-check cutouts from MSSQL  [C# — Ormeo.Document]
─────────────────────────────────────────────────
Query TesseractLineImage table, extract varbinary + validated text
  → training_data/spot_check/  (40K verified line_image + label pairs)

Step 3: Train detection  [Python — training/detect/]
───────────────────────
Detection bbox labels come from Step 1 (Tesseract TSV bboxes) + Ormeo layout pipeline
python training/detect/train.py --config detect_config.yaml \
    --data training_data/detection/

Step 4a: Train PARSeq recognition  [Python — training/recognize/]
──────────────────────────────────
python training/recognize/train_parseq.py \
    --auto-labeled D:/TrainingData/splits/train \
    --spot-check D:/TrainingData/spot_check \
    --output-dir output/parseq_nordic
  Data: tesseract_auto 200K (weight 0.5) + spot_check 40K×4 (weight 0.4)
  → output/parseq_nordic/best.pth

Step 4b: Train SVTRv2 recognition  [Python — training/recognize/]
──────────────────────────────────
# First: convert TSV+images → LMDB (required by OpenOCR)
python training/tools/tsv_to_lmdb.py \
    --input D:/TrainingData/splits/train \
    --output D:/TrainingData/lmdb/train \
    --charset-file training/recognize/nordic_dict.txt

python training/tools/tsv_to_lmdb.py \
    --input D:/TrainingData/splits/val \
    --output D:/TrainingData/lmdb/val \
    --charset-file training/recognize/nordic_dict.txt

# Then: train via OpenOCR
python training/recognize/train_svtrv2.py \
    --train-lmdb D:/TrainingData/lmdb/train \
    --val-lmdb D:/TrainingData/lmdb/val \
    --pretrained svtrv2_rctc \
    --output-dir output/svtrv2_nordic \
    --variant base --decoder rctc
  → output/svtrv2_nordic/best.pth

Step 5: Evaluate  [Python — training/eval/]
────────────────
# PARSeq evaluation
python training/eval/evaluate.py \
    --model parseq \
    --checkpoint output/parseq_nordic/best.pth \
    --test-dir D:/TrainingData/splits/test \
    --output-dir output/parseq_nordic/eval

# SVTRv2 evaluation (same metrics, CTC decoding)
python training/eval/evaluate.py \
    --model svtrv2 \
    --checkpoint output/svtrv2_nordic/best.pth \
    --test-dir D:/TrainingData/splits/test \
    --output-dir output/svtrv2_nordic/eval

# Outputs: metrics.json (CER, WER, diacritical accuracy, confusion matrix),
#          predictions.tsv (every prediction), worst_50.tsv (manual review)

Step 6: Iterate if needed  [Python + optionally C# for more data]
─────────────────────────
If diacritical accuracy < 99%:
  - Raise confidence threshold to ≥ 98 (re-run C# export with stricter filter)
  - Increase cutout oversampling to 10×
  - Add VLM-labeled data for moderate-confidence Tesseract outputs
  - Add targeted synthetic for specific failing diacritical pairs
Re-train and re-evaluate

Step 7: Export and build engines  [Python + trtexec on target GPU]
────────────────────────────────
python training/export/export_onnx_all.py \
    --checkpoints-dir checkpoints/ --output-dir models/
```

---

## Hardware Requirements

**GPU is required for model training.** PARSeq, SVTRv2, DBNet++, and FAST are transformer/CNN models — training on CPU is impractical (weeks vs hours).

Data preparation (Step 1-2) is CPU/IO-bound and needs no GPU.

### Per-step requirements

| Step | Where | Hardware | VRAM | Time estimate |
|---|---|---|---|---|
| Tesseract auto-labeling (~5K-10K pages → 200K crops) | C# (Ormeo.Document) | CPU (multi-core) | — | ~1-4 hours |
| Export 40K cutouts from MSSQL | C# (Ormeo.Document) | CPU | — | ~minutes |
| Recognition training (200K samples, 20 epochs) | Python (training/) | **GPU required** | ~8-12 GB | ~4-8 hours (A6000 Ada) |
| Detection fine-tuning (5K-10K pages, 100 epochs) | Python (training/) | **GPU required** | ~16-24 GB | ~4-8 hours (A6000 Ada) |
| (Optional) Synthetic generation | Python (training/) | CPU | — | ~1-2 hours for 50K |
| (Optional) VLM labeling | Python or C# | GPU | ~16-64 GB | ~hours |
| TensorRT engine build | trtexec CLI | **Target GPU** | varies | ~10-30 min/engine |

### GPU sizing

| Task | Minimum GPU | Recommended | Notes |
|---|---|---|---|
| Recognition fine-tuning (PARSeq/SVTRv2) | 8 GB VRAM | 2× RTX 6000 PRO Blackwell | ~1-3 hours with 2-GPU data-parallel |
| Detection fine-tuning (DBNet++/FAST) | 16 GB VRAM | 2× RTX 6000 PRO Blackwell | ~1-3 hours with 2-GPU data-parallel |
| TensorRT engine build (sm_89) | A6000 Ada | — | Must build on the target GPU |
| TensorRT engine build (sm_120) | RTX 6000 PRO Blackwell | — | Must build on the target GPU |

### Training vs engine build — different machines

Training produces a GPU-agnostic PyTorch checkpoint → ONNX file. Train on the fastest GPU available (Blackwell). TensorRT engines must be built on each deployment target GPU separately.

```
Train on 2× RTX 6000 PRO Blackwell (fastest available)
   → checkpoint.pth → model.onnx  (GPU-agnostic)
        │
        ├──► trtexec on A6000 Ada         → model_sm89.engine
        └──► trtexec on RTX 6000 Blackwell → model_sm120.engine
```

### What does NOT need a GPU

The C# data preparation steps are I/O and CPU bound (SeaweedFS download + Tesseract). These run on any machine with:
- Network access to SeaweedFS
- Tesseract installed
- Enough disk for 200K PNG images (~20-50 GB)
- MSSQL access for the 40K spot-check cutouts

The existing Ormeo.Document pipeline already handles parallel processing.

---

## VLM-Based OCR Models: Assessment for nordocr

Several VLM-based OCR models were evaluated as potential alternatives or complements to the nordocr pipeline. None replace the TensorRT inference engine, but some may be useful as labeling oracles or spot-check layers.

### Why VLMs cannot replace nordocr's TensorRT pipeline

| Characteristic | nordocr (PARSeq/SVTRv2 on TensorRT) | VLM-based OCR |
|---|---|---|
| Architecture | Feed-forward encoder, parallel decode | Autoregressive token generation |
| Serving | Static ONNX → TensorRT engine | vLLM / TensorRT-LLM |
| Per-page latency | Target: <50ms | 200-1500ms |
| Per-line throughput | Thousands/sec | N/A (page-level only) |
| Export | ONNX → TensorRT (trivial) | No static graph export |
| Custom charset | Output head swap (simple) | LoRA fine-tune (heavier) |
| VRAM at inference | ~2-4 GB (20M param model) | 16-64 GB (0.9B-32B models) |

The throughput gap is fundamental and architectural — autoregressive generation cannot match parallel feed-forward inference for per-character recognition speed.

### GLM-OCR (Zhipu AI, February 2026)

| Property | Value |
|---|---|
| Parameters | 0.9B |
| Architecture | CogViT encoder + GLM-0.5B decoder |
| License | MIT |
| Throughput | ~0.67 images/sec (page-level) |
| Accuracy | #1 on OmniDocBench v1.5 (94.62) |
| Languages | Effectively English/Chinese only |
| ONNX/TensorRT | No — autoregressive VLM, requires vLLM |
| Nordic support | **Poor** — independent testing shows failures on non-English/Chinese text |

**Verdict: Not useful for Nordic documents.** Despite impressive benchmark scores on English/Chinese, independent testing reports failures ("hung or printed gibberish") on other languages. Nordic diacriticals (å, ä, ö, ø, æ) would almost certainly fail without heavy fine-tuning, and even then the language coverage is uncertain. The MIT license is favorable but the language limitation is a dealbreaker.

### DeepSeek-OCR / DeepSeek-OCR-2 (DeepSeek AI, October 2025 / January 2026)

| Property | Value |
|---|---|
| Parameters | 3B total (MoE), 570M activated |
| Architecture | DeepEncoder (SAM + CLIP) + DeepSeek-3B-MoE decoder |
| License | MIT (v1), Apache 2.0 (v2) |
| Throughput | ~2-3 pages/sec on A100, 200K+ pages/day |
| Accuracy | Beats GOT-OCR 2.0 with 100 vision tokens vs 256 |
| Languages | Reportedly 100+ (Latin, CJK, Cyrillic, Arabic) — Nordic unverified |
| ONNX/TensorRT | No — autoregressive MoE decoder, requires vLLM |
| Nordic support | **Unverified** — Latin diacriticals should work architecturally but no benchmarks exist |
| Fine-tuning | Well-supported via Unsloth/LoRA (Persian fine-tune: 88% CER improvement in 60 steps) |
| Hardware | Runs on A6000 Ada (48GB) comfortably |

**Verdict: Potentially useful as labeling oracle or spot-check replacement, not as primary OCR.**

DeepSeek-OCR-2 is interesting for two specific roles in the nordocr pipeline:

#### Potential role 1: Replace Qwen3-VL 32B as spot-check layer

| | Qwen3-VL 32B (current) | DeepSeek-OCR-2 (3B) |
|---|---|---|
| Active parameters | 32B | 570M |
| VRAM needed | ~64GB+ | ~16-24GB |
| Speed per page | Slower | ~2-3 pages/sec |
| OCR-specialized | No (general VLM) | Yes (trained specifically for document OCR) |
| Nordic verified | Yes (broad multilingual) | No — needs evaluation |

If DeepSeek-OCR-2 handles Nordic text well (after optional LoRA fine-tuning), it could replace Qwen3-VL 32B as the spot-check layer at ~10x less compute cost. This is significant if spot-checking runs on every page.

**Prerequisite**: Evaluate DeepSeek-OCR-2 on 100+ Nordic TIFF pages and compare diacritical accuracy against Qwen3-VL 32B before committing.

#### Potential role 2: Faster labeling oracle for training data

The training pipeline (Phase 2) currently plans to use Qwen3-VL 32B to label 50K real TIFF pages. DeepSeek-OCR-2 could do this faster and cheaper:

| | Qwen3-VL 32B | DeepSeek-OCR-2 |
|---|---|---|
| Labeling 50K pages | ~24-48 hours on A6000 | ~6-12 hours on A6000 |
| VRAM during labeling | Tight on 48GB | Comfortable on 48GB |
| Can run alongside training | Unlikely (VRAM) | Possible |

**Open question**: Whether DeepSeek-OCR-2's Nordic accuracy is good enough for silver-standard ground truth labels. This needs empirical testing before adopting it as the labeling oracle.

### Summary: VLM role in nordocr

```
Production pipeline:
  Scan → Ormeo preprocess → nordocr detect+recognize (TensorRT) → text
                                         │
                                    Fast, parallel,
                                    <50ms/page target
                                         │
                            Optional: spot-check layer ← DeepSeek-OCR-2? (evaluate first)
                                                         or Qwen3-VL 32B (current)

Training pipeline:
  Real TIFFs → labeling oracle → silver ground truth → fine-tune nordocr models
                    │
               DeepSeek-OCR-2? (evaluate first)
               or Qwen3-VL 32B (current, proven)
```

The VLMs serve as quality layers (labeling, verification), not as the primary OCR engine. nordocr's TensorRT pipeline handles throughput; VLMs handle accuracy assurance where speed is less critical.

### Open questions (to be resolved by evaluation)

1. **Does DeepSeek-OCR-2 handle Nordic diacriticals reliably?** Test on 100+ pages with known ground truth.
2. **If not out-of-the-box, does LoRA fine-tuning fix it?** The Persian fine-tuning results (88% CER improvement) suggest it could adapt quickly.
3. **Is DeepSeek-OCR-2's accuracy on Nordic text close enough to Qwen3-VL 32B?** If within 1-2% CER, the 10x compute savings make it worthwhile.
4. **Can DeepSeek-OCR-2 run alongside nordocr training on the same A6000 Ada?** At 570M active params / ~16GB VRAM, this should be feasible if training batch sizes are adjusted.
