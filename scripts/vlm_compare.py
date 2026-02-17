"""
Compare OCR results against Qwen3 VL 30B as independent ground truth.

Crops each detected region from page PNGs, sends to VLM for reading,
then compares VLM text against C# Tesseract results.

Usage:
    python scripts/vlm_compare.py
"""

import base64
import io
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

VLM_URL = "http://10.12.2.31:8001/v1/chat/completions"
VLM_MODEL = "Qwen3 VL 30B A3B"
PAGES_DIR = Path(r"c:\temp\tiff_pages")
CSHARP_JSON = Path(r"C:\Temp\tesseract-pipeline\ocr_results.json")
OUTPUT_JSON = Path(r"C:\Temp\vlm_comparison.json")
MAX_WORKERS = 1  # sequential requests


def load_pages():
    """Load all page PNGs, sorted by name."""
    pngs = sorted(PAGES_DIR.glob("page_*.png"))
    pages = {}
    for p in pngs:
        # Extract page number from filename (page_00.png -> 0)
        num = int(p.stem.split("_")[1])
        pages[num] = Image.open(p).convert("RGB")
    print(f"Loaded {len(pages)} pages")
    return pages


def crop_region(page_img, region):
    """Crop a region from a page image with bounds clamping."""
    x = max(0, region["x"])
    y = max(0, region["y"])
    w = region["width"]
    h = region["height"]
    pw, ph = page_img.size

    x2 = min(x + w, pw)
    y2 = min(y + h, ph)

    if x2 <= x or y2 <= y:
        return None

    return page_img.crop((x, y, x2, y2))


def crop_to_base64(crop_img):
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    crop_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_vlm(b64_image, timeout=30):
    """Send a cropped image to the VLM and get text back."""
    resp = requests.post(
        VLM_URL,
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Read the text in this image exactly as written. "
                                "Output ONLY the raw text, nothing else. "
                                "No quotes, no explanation, no markdown."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                        },
                    ],
                }
            ],
            "max_tokens": 300,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def normalize_text(s):
    """Normalize text for comparison (collapse whitespace, strip)."""
    return " ".join(s.split()).strip()


def process_region(idx, region, pages):
    """Process a single region: crop, VLM query, return result."""
    page_num = region["pageNumber"]
    page_img = pages.get(page_num)
    if page_img is None:
        return idx, {"error": f"page {page_num} not loaded"}

    crop = crop_region(page_img, region)
    if crop is None:
        return idx, {"error": "invalid crop bounds"}

    b64 = crop_to_base64(crop)

    try:
        vlm_text = query_vlm(b64)
    except Exception as e:
        return idx, {"error": str(e)}

    cs_text = region["text"]
    cs_norm = normalize_text(cs_text)
    vlm_norm = normalize_text(vlm_text)

    exact = cs_norm == vlm_norm
    case_match = cs_norm.lower() == vlm_norm.lower()

    return idx, {
        "page": page_num,
        "row": region["row"],
        "col": region["column"],
        "bbox": [region["x"], region["y"], region["width"], region["height"]],
        "cs_text": cs_text,
        "vlm_text": vlm_text,
        "exact": exact,
        "case_match": case_match,
    }


def main():
    if not PAGES_DIR.exists() or not CSHARP_JSON.exists():
        print(f"Need {PAGES_DIR} and {CSHARP_JSON}")
        sys.exit(1)

    pages = load_pages()

    with open(CSHARP_JSON, encoding="utf-8") as f:
        regions = json.load(f)
    print(f"C# regions: {len(regions)}")

    results = [None] * len(regions)
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_region, i, r, pages): i
            for i, r in enumerate(regions)
        }

        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result
            done += 1

            if done % 50 == 0 or done == len(regions):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(regions) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(regions)}] {elapsed:.0f}s elapsed, "
                    f"{rate:.1f} regions/s, ETA {eta:.0f}s"
                )

    # Save full results.
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # Summary statistics.
    errors = sum(1 for r in results if r and "error" in r)
    valid = [r for r in results if r and "error" not in r]
    exact = sum(1 for r in valid if r["exact"])
    case = sum(1 for r in valid if r["case_match"] and not r["exact"])
    diff = sum(1 for r in valid if not r["case_match"])

    print(f"\n=== VLM vs C# Tesseract Comparison ===")
    print(f"Total regions:  {len(regions)}")
    print(f"Errors:         {errors}")
    print(f"Exact match:    {exact} ({exact/len(valid)*100:.1f}%)")
    print(f"Case match:     {case} ({case/len(valid)*100:.1f}%)")
    print(f"Different:      {diff} ({diff/len(valid)*100:.1f}%)")

    # Per-page summary.
    from collections import defaultdict

    page_stats = defaultdict(lambda: {"total": 0, "exact": 0, "diff": 0})
    for r in valid:
        ps = page_stats[r["page"]]
        ps["total"] += 1
        if r["exact"]:
            ps["exact"] += 1
        elif not r["case_match"]:
            ps["diff"] += 1

    print(f"\n{'Page':>5}  {'Total':>6}  {'Exact':>6}  {'Diff':>6}  {'%Exact':>7}")
    print("-" * 40)
    for p in sorted(page_stats.keys()):
        ps = page_stats[p]
        pct = ps["exact"] / ps["total"] * 100 if ps["total"] else 0
        print(f"{p:5}  {ps['total']:6}  {ps['exact']:6}  {ps['diff']:6}  {pct:6.1f}%")

    # Show differences.
    print(f"\n=== Differences (VLM != C#) ===")
    diffs = [r for r in valid if not r["case_match"]]
    for r in diffs[:50]:
        cs = r["cs_text"][:60]
        vlm = r["vlm_text"][:60]
        prefix = f"p{r['page']} row{r['row']}"
        print(f"  {prefix}: C#=\"{cs}\"")
        print(f"  {' '*len(prefix)}  VLM=\"{vlm}\"")


if __name__ == "__main__":
    main()
