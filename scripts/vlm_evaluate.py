"""
Evaluate nordocr OCR quality using Qwen3 VL 30B as independent referee.

Runs a TIFF through the nordocr pipeline, then sends each detected region
to the VLM for independent reading. Compares nordocr output vs VLM text
and reports accuracy metrics (exact match %, CER).

Usage:
    python scripts/vlm_evaluate.py path/to/document.tiff [--output results.json] [--workers 4]
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

# --- Constants ---

VLM_URL = "http://10.12.2.31:8001/v1/chat/completions"
VLM_MODEL = "Qwen3 VL 30B A3B"

# DLL paths for running nordocr on dev machine (Windows)
TRT_LIBS = (
    r"C:\Users\Jens Nylander\AppData\Local\Packages"
    r"\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python312\site-packages\tensorrt_libs"
)
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64"


# --- Helpers ---


def normalize_text(s):
    """Collapse whitespace and strip for comparison."""
    return " ".join(s.split()).strip()


def char_distance(a, b):
    """Levenshtein distance (Wagner-Fischer)."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def cer(ref, hyp):
    """Character Error Rate: edit_distance / len(ref)."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return char_distance(ref, hyp) / len(ref)


def crop_region(page_img, bbox):
    """Crop a region from a page image using bbox dict {x, y, width, height}."""
    x = max(0, int(bbox["x"]))
    y = max(0, int(bbox["y"]))
    w = int(bbox["width"])
    h = int(bbox["height"])
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


def query_vlm(b64_image, timeout=30, retries=1):
    """Send a cropped image to the VLM and get text back. Retries on timeout."""
    for attempt in range(1 + retries):
        try:
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
                                    "image_url": {
                                        "url": f"data:image/png;base64,{b64_image}"
                                    },
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
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt < retries:
                time.sleep(2)
                continue
            raise


# --- Pipeline helpers ---


def find_repo_root(start=None):
    """Walk up from start dir to find repo root (has Cargo.toml with [workspace])."""
    d = Path(start or __file__).resolve().parent
    while d != d.parent:
        cargo = d / "Cargo.toml"
        if cargo.exists() and "[workspace]" in cargo.read_text(encoding="utf-8"):
            return d
        d = d.parent
    return None


def find_nordocr_binary(repo_root):
    """Find the nordocr release binary."""
    for name in ["nordocr.exe", "nordocr"]:
        p = repo_root / "target" / "release" / name
        if p.exists():
            return p
    return None


def find_engine_path(repo_root):
    """Find the best available recognition engine file."""
    models = repo_root / "models"
    # Prefer sm86 (Ampere dev machine), then sm89, then sm120
    for suffix in ["sm86", "sm89", "sm120"]:
        p = models / f"recognize_svtrv2_{suffix}.engine"
        if p.exists():
            return str(p)
    # Fallback: any engine file
    if models.exists():
        for f in models.iterdir():
            if f.name.startswith("recognize") and f.suffix == ".engine":
                return str(f)
    return None


def decode_tiff_pages(tiff_path):
    """Decode a (possibly multi-page) TIFF into a list of RGB PIL Images."""
    img = Image.open(tiff_path)
    pages = []
    try:
        while True:
            pages.append(img.convert("RGB"))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return pages


def build_config_json(repo_root):
    """Write a minimal pipeline config JSON, return path to temp file."""
    engine = find_engine_path(repo_root)
    if not engine:
        print("ERROR: No recognition engine found in models/", file=sys.stderr)
        print("  Build one with: python training/recognize/build_trt_engine.py", file=sys.stderr)
        sys.exit(1)

    config = {
        "recognize_engine_path": engine,
        "detect_model": "Morphological",
        "recognize_model": "SVTRv2",
        "precision": "FP16",
        "detect_max_batch": 2,
        "detect_input_height": 1024,
        "detect_input_width": 1024,
        "detect_threshold": 0.3,
        "detect_min_area": 100.0,
        "recognize_max_batch": 32,
        "recognize_input_height": 32,
        "recognize_max_input_width": 1792,
        "recognize_max_seq_len": 448,
        "num_streams": 2,
        "gpu_pool_size": 268435456,
        "enable_cuda_graph": True,
        "enable_preprocess": True,
        "enable_dali": False,
        "dla": {
            "offload_detect": False,
            "offload_recognize": False,
            "dla_core": 0,
            "allow_gpu_fallback": True,
        },
        "gpu_arch_override": "sm_86",
    }

    fd, path = tempfile.mkstemp(suffix=".json", prefix="nordocr_config_")
    with os.fdopen(fd, "w") as f:
        json.dump(config, f, indent=2)
    return path


def build_env():
    """Build environment dict with TRT and CUDA DLLs on PATH."""
    env = os.environ.copy()
    extra = []
    if os.path.isdir(TRT_LIBS):
        extra.append(TRT_LIBS)
    if os.path.isdir(CUDA_BIN):
        extra.append(CUDA_BIN)
    if extra:
        env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
    return env


def run_nordocr_single(binary, config_path, image_path, env):
    """Run nordocr on a single image file, return parsed JSON output or None on error."""
    cmd = [
        str(binary),
        "--config", config_path,
        "--log-level", "error",
        "process",
        str(image_path),
        "--format", "json",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        timeout=300,
    )

    if result.returncode != 0:
        print(f"  WARNING: nordocr failed on {Path(image_path).name}: {result.stderr.strip()}", file=sys.stderr)
        return None

    stdout = result.stdout.strip()
    if not stdout:
        print(f"  WARNING: nordocr produced no output for {Path(image_path).name}", file=sys.stderr)
        if result.stderr.strip():
            print(f"    stderr: {result.stderr.strip()[:200]}", file=sys.stderr)
        return None

    # Extract JSON from output (tracing logs may be mixed into stdout).
    # Find the first '{' that starts a JSON object.
    json_start = stdout.find("{")
    if json_start < 0:
        print(f"  WARNING: no JSON found in nordocr output for {Path(image_path).name}", file=sys.stderr)
        return None

    try:
        return json.loads(stdout[json_start:])
    except json.JSONDecodeError as e:
        print(f"  WARNING: invalid JSON from nordocr: {e}", file=sys.stderr)
        return None


def run_nordocr(pages, repo_root):
    """Run nordocr pipeline on pre-decoded PIL pages.

    Converts each page to a temp PNG (avoids TIFF Fax4 incompatibility with
    the tiff crate), runs nordocr on each, and merges results.
    """
    binary = find_nordocr_binary(repo_root)
    if not binary:
        print("ERROR: nordocr binary not found.", file=sys.stderr)
        print("  Build with: cargo build --release", file=sys.stderr)
        sys.exit(1)

    config_path = build_config_json(repo_root)
    env = build_env()
    all_pages = []
    timing_totals = {"decode_ms": 0, "preprocess_ms": 0, "detect_ms": 0, "recognize_ms": 0, "total_ms": 0}

    try:
        for page_idx, page_img in enumerate(pages):
            # Save page as temp PNG.
            fd, png_path = tempfile.mkstemp(suffix=".png", prefix=f"nordocr_page{page_idx}_")
            os.close(fd)
            try:
                page_img.save(png_path, format="PNG")
                print(f"  Processing page {page_idx}...", file=sys.stderr)
                output = run_nordocr_single(binary, config_path, png_path, env)
                if output and output.get("pages"):
                    # Re-index page to match original page number.
                    for p in output["pages"]:
                        p["page_index"] = page_idx
                    all_pages.extend(output["pages"])
                    timing = output.get("timing", {})
                    for k in timing_totals:
                        timing_totals[k] += timing.get(k, 0)
            finally:
                os.unlink(png_path)

    finally:
        os.unlink(config_path)

    return {"pages": all_pages, "timing": timing_totals}


# --- Evaluation ---


def evaluate_region(page_img, line, page_idx):
    """Evaluate a single recognized region against VLM."""
    bbox = line["bbox"]
    crop = crop_region(page_img, bbox)
    if crop is None:
        return {"error": "invalid crop bounds", "page": page_idx}

    b64 = crop_to_base64(crop)

    try:
        vlm_text = query_vlm(b64)
    except Exception as e:
        return {
            "error": str(e),
            "page": page_idx,
            "nordocr_text": line["text"],
            "bbox": bbox,
        }

    nordocr_norm = normalize_text(line["text"])
    vlm_norm = normalize_text(vlm_text)

    exact = nordocr_norm == vlm_norm
    case_match = nordocr_norm.lower() == vlm_norm.lower()
    c = cer(vlm_norm, nordocr_norm)  # VLM as reference

    return {
        "page": page_idx,
        "bbox": bbox,
        "nordocr_text": line["text"],
        "vlm_text": vlm_text,
        "confidence": line.get("confidence", 0.0),
        "exact": exact,
        "case_match": case_match,
        "cer": round(c, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nordocr OCR quality using VLM as referee"
    )
    parser.add_argument("input", help="Path to TIFF file")
    parser.add_argument(
        "--output", "-o", help="Output JSON path (default: <input_stem>_vlm_eval.json)"
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="VLM query workers (default: 4)"
    )
    args = parser.parse_args()

    tiff_path = Path(args.input).resolve()
    if not tiff_path.exists():
        print(f"ERROR: {tiff_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(tiff_path.with_suffix("")) + "_vlm_eval.json"

    # Find repo root.
    repo_root = find_repo_root()
    if not repo_root:
        print("ERROR: Could not find nordocr repo root (Cargo.toml with [workspace])", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {tiff_path}", file=sys.stderr)
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"Repo:   {repo_root}", file=sys.stderr)

    # 1. Decode TIFF pages for cropping.
    print("\nDecoding TIFF pages...", file=sys.stderr)
    pages = decode_tiff_pages(tiff_path)
    print(f"  {len(pages)} page(s)", file=sys.stderr)

    # 2. Run nordocr pipeline (page-by-page via temp PNGs to avoid Fax4 issues).
    print("\nRunning nordocr pipeline...", file=sys.stderr)
    nordocr_output = run_nordocr(pages, repo_root)
    timing = nordocr_output.get("timing", {})
    page_results = nordocr_output.get("pages", [])
    total_lines = sum(len(p.get("lines", [])) for p in page_results)
    print(
        f"  {len(page_results)} page(s), {total_lines} regions, "
        f"{timing.get('total_ms', 0):.0f}ms",
        file=sys.stderr,
    )

    # 3. Build evaluation tasks.
    tasks = []
    for page in page_results:
        page_idx = page["page_index"]
        if page_idx >= len(pages):
            print(
                f"  WARNING: page_index {page_idx} but only {len(pages)} images",
                file=sys.stderr,
            )
            continue
        page_img = pages[page_idx]
        for line in page.get("lines", []):
            tasks.append((page_img, line, page_idx))

    if not tasks:
        print("No regions to evaluate.", file=sys.stderr)
        sys.exit(0)

    # 4. Query VLM for each region.
    print(f"\nQuerying VLM ({len(tasks)} regions, {args.workers} workers)...", file=sys.stderr)
    results = [None] * len(tasks)
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(evaluate_region, img, line, pidx): i
            for i, (img, line, pidx) in enumerate(tasks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            if done % 20 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{len(tasks)}] {elapsed:.0f}s, "
                    f"{rate:.1f} regions/s, ETA {eta:.0f}s",
                    file=sys.stderr,
                )

    # 5. Save results.
    output_data = {
        "input": str(tiff_path),
        "timing": timing,
        "regions": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}", file=sys.stderr)

    # 6. Report.
    errors = [r for r in results if r and "error" in r]
    valid = [r for r in results if r and "error" not in r]

    if not valid:
        print("No valid results to report.", file=sys.stderr)
        sys.exit(0)

    exact_count = sum(1 for r in valid if r["exact"])
    case_count = sum(1 for r in valid if r["case_match"] and not r["exact"])
    diff_count = sum(1 for r in valid if not r["case_match"])
    avg_cer = sum(r["cer"] for r in valid) / len(valid)

    print(f"\n{'='*50}", file=sys.stderr)
    print(f"  nordocr vs VLM Evaluation", file=sys.stderr)
    print(f"{'='*50}", file=sys.stderr)
    print(f"  Total regions:  {len(results)}", file=sys.stderr)
    print(f"  VLM errors:     {len(errors)}", file=sys.stderr)
    print(f"  Exact match:    {exact_count}/{len(valid)} ({exact_count/len(valid)*100:.1f}%)", file=sys.stderr)
    print(f"  Case match:     {case_count}/{len(valid)} ({case_count/len(valid)*100:.1f}%)", file=sys.stderr)
    print(f"  Different:      {diff_count}/{len(valid)} ({diff_count/len(valid)*100:.1f}%)", file=sys.stderr)
    print(f"  Average CER:    {avg_cer:.4f} ({avg_cer*100:.2f}%)", file=sys.stderr)
    print(f"  Pipeline time:  {timing.get('total_ms', 0):.0f}ms", file=sys.stderr)

    # Per-page summary.
    from collections import defaultdict

    page_stats = defaultdict(lambda: {"total": 0, "exact": 0, "cer_sum": 0.0})
    for r in valid:
        ps = page_stats[r["page"]]
        ps["total"] += 1
        ps["cer_sum"] += r["cer"]
        if r["exact"]:
            ps["exact"] += 1

    print(f"\n  {'Page':>5}  {'Regions':>8}  {'Exact%':>7}  {'Avg CER':>8}", file=sys.stderr)
    print(f"  {'-'*35}", file=sys.stderr)
    for p in sorted(page_stats.keys()):
        ps = page_stats[p]
        pct = ps["exact"] / ps["total"] * 100 if ps["total"] else 0
        ac = ps["cer_sum"] / ps["total"] if ps["total"] else 0
        print(f"  {p:5}  {ps['total']:8}  {pct:6.1f}%  {ac:7.4f}", file=sys.stderr)

    # Show differences (first 50).
    diffs = [r for r in valid if not r["case_match"]]
    if diffs:
        print(f"\n  Differences ({len(diffs)} total, showing first 50):", file=sys.stderr)
        for r in diffs[:50]:
            nord = r["nordocr_text"][:60]
            vlm = r["vlm_text"][:60]
            print(
                f"    p{r['page']} CER={r['cer']:.2f} conf={r['confidence']:.2f}: "
                f'nordocr="{nord}"',
                file=sys.stderr,
            )
            print(f"      {'':>30} VLM=\"{vlm}\"", file=sys.stderr)


if __name__ == "__main__":
    main()
