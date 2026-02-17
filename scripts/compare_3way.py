"""
3-way comparison: SVTRv2 vs Tesseract vs VLM (Qwen3 VL 30B).

Loads pre-computed JSON results from both OCR backends (same bboxes),
crops each region from the source TIFF, queries the VLM once per region,
and compares all three.

Usage:
    python scripts/compare_3way.py C:\Temp\06313923_all.tif \
        --svtrv2 C:\Temp\svtrv2_result.json \
        --tess C:\Temp\tess_result.json \
        --workers 8
"""

import argparse
import base64
import io
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

VLM_URL = "http://10.12.2.31:8001/v1/chat/completions"
VLM_MODEL = "Qwen3 VL 30B A3B"


def normalize(s):
    return " ".join(s.split()).strip()


def levenshtein(a, b):
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def cer(ref, hyp):
    if len(ref) == 0: return 0.0 if len(hyp) == 0 else 1.0
    return levenshtein(ref, hyp) / len(ref)


def crop_to_b64(page_img, bbox):
    x, y = max(0, int(bbox["x"])), max(0, int(bbox["y"]))
    w, h = int(bbox["width"]), int(bbox["height"])
    pw, ph = page_img.size
    crop = page_img.crop((x, y, min(x+w, pw), min(y+h, ph)))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def query_vlm(b64_img, timeout=30):
    for attempt in range(2):
        try:
            r = requests.post(VLM_URL, json={
                "model": VLM_MODEL,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text":
                        "Read the text in this image exactly as written. "
                        "Output ONLY the raw text, nothing else. "
                        "No quotes, no explanation, no markdown."},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_img}"}}
                ]}],
                "max_tokens": 300,
                "temperature": 0.0,
            }, timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 0:
                time.sleep(2)
                continue
            return "[VLM_TIMEOUT]"
    return "[VLM_ERROR]"


def process_region(page_img, bbox, svtrv2_text, tess_text, page_idx, region_idx):
    b64 = crop_to_b64(page_img, bbox)
    vlm_text = query_vlm(b64)
    vlm_n = normalize(vlm_text)
    svt_n = normalize(svtrv2_text)
    tss_n = normalize(tess_text)

    return {
        "idx": region_idx,
        "page": page_idx,
        "bbox": bbox,
        "svtrv2": svtrv2_text,
        "tess": tess_text,
        "vlm": vlm_text,
        "svt_exact": svt_n == vlm_n,
        "tss_exact": tss_n == vlm_n,
        "svt_space": svt_n.replace(" ", "") == vlm_n.replace(" ", ""),
        "tss_space": tss_n.replace(" ", "") == vlm_n.replace(" ", ""),
        "svt_cer": round(cer(vlm_n, svt_n), 4),
        "tss_cer": round(cer(vlm_n, tss_n), 4),
    }


def main():
    ap = argparse.ArgumentParser(description="3-way OCR comparison: SVTRv2 vs Tesseract vs VLM")
    ap.add_argument("tiff", help="Source TIFF (LZW-converted)")
    ap.add_argument("--svtrv2", required=True, help="SVTRv2 JSON result")
    ap.add_argument("--tess", required=True, help="Tesseract JSON result")
    ap.add_argument("--output", "-o", default=None, help="Output JSON path")
    ap.add_argument("--workers", "-w", type=int, default=8)
    args = ap.parse_args()

    # Load inputs.
    with open(args.svtrv2, encoding="utf-8") as f:
        svtrv2 = json.load(f)
    with open(args.tess, encoding="utf-8") as f:
        tess = json.load(f)

    img = Image.open(args.tiff)
    pages = []
    try:
        while True:
            pages.append(img.convert("RGB").copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass

    print(f"Loaded {len(pages)} pages", file=sys.stderr)

    # Build task list from SVTRv2 results (same bboxes as Tesseract).
    tasks = []
    for sp, tp in zip(svtrv2["pages"], tess["pages"]):
        page_idx = sp["page_index"]
        assert page_idx == tp["page_index"]
        assert len(sp["lines"]) == len(tp["lines"])
        page_img = pages[page_idx]
        for sl, tl in zip(sp["lines"], tp["lines"]):
            tasks.append((page_img, sl["bbox"], sl["text"], tl["text"], page_idx))

    print(f"{len(tasks)} regions to evaluate with {args.workers} workers", file=sys.stderr)

    # Query VLM in parallel.
    results = [None] * len(tasks)
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_region, img, bbox, svt, tss, pidx, i): i
            for i, (img, bbox, svt, tss, pidx) in enumerate(tasks)
        }
        for f in as_completed(futures):
            idx = futures[f]
            try:
                results[idx] = f.result()
            except Exception as e:
                results[idx] = {"idx": idx, "error": str(e)}
            done += 1
            if done % 50 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - done) / rate if rate > 0 else 0
                print(f"  [{done}/{len(tasks)}] {elapsed:.0f}s, {rate:.1f} rgn/s, ETA {eta:.0f}s",
                      file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nVLM queries done in {elapsed:.0f}s", file=sys.stderr)

    # Filter valid results.
    valid = [r for r in results if r and "error" not in r]
    errors = [r for r in results if r and "error" in r]
    vlm_empty = [r for r in valid if not normalize(r["vlm"]).strip()]

    # Summary stats.
    svt_exact = sum(1 for r in valid if r["svt_exact"])
    tss_exact = sum(1 for r in valid if r["tss_exact"])
    svt_space = sum(1 for r in valid if r["svt_space"])
    tss_space = sum(1 for r in valid if r["tss_space"])
    svt_cer_avg = sum(r["svt_cer"] for r in valid) / len(valid) if valid else 0
    tss_cer_avg = sum(r["tss_cer"] for r in valid) / len(valid) if valid else 0
    n = len(valid)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  3-Way Comparison: SVTRv2 vs Tesseract vs VLM", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Total regions:     {len(results)}", file=sys.stderr)
    print(f"  VLM errors:        {len(errors)}", file=sys.stderr)
    print(f"  VLM empty:         {len(vlm_empty)}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"  {'':>20} {'SVTRv2':>10} {'Tesseract':>10}", file=sys.stderr)
    print(f"  {'-'*42}", file=sys.stderr)
    print(f"  {'Exact match':>20} {svt_exact:>6}/{n} {tss_exact:>6}/{n}", file=sys.stderr)
    print(f"  {'Exact %':>20} {svt_exact/n*100:>9.1f}% {tss_exact/n*100:>9.1f}%", file=sys.stderr)
    print(f"  {'Space-insensitive':>20} {svt_space:>6}/{n} {tss_space:>6}/{n}", file=sys.stderr)
    print(f"  {'Space-insensitive %':>20} {svt_space/n*100:>9.1f}% {tss_space/n*100:>9.1f}%", file=sys.stderr)
    print(f"  {'Avg CER':>20} {svt_cer_avg*100:>9.2f}% {tss_cer_avg*100:>9.2f}%", file=sys.stderr)

    # Per-page breakdown.
    ps = defaultdict(lambda: {"n": 0, "svt_ex": 0, "tss_ex": 0, "svt_cer": 0.0, "tss_cer": 0.0})
    for r in valid:
        p = ps[r["page"]]
        p["n"] += 1
        p["svt_cer"] += r["svt_cer"]
        p["tss_cer"] += r["tss_cer"]
        if r["svt_exact"]: p["svt_ex"] += 1
        if r["tss_exact"]: p["tss_ex"] += 1

    print(f"\n  {'Page':>5} {'Rgns':>5} | {'SVT exact%':>10} {'SVT CER':>8} | {'TSS exact%':>10} {'TSS CER':>8}", file=sys.stderr)
    print(f"  {'-'*60}", file=sys.stderr)
    for p in sorted(ps.keys()):
        s = ps[p]
        svt_pct = s["svt_ex"]/s["n"]*100 if s["n"] else 0
        tss_pct = s["tss_ex"]/s["n"]*100 if s["n"] else 0
        svt_c = s["svt_cer"]/s["n"]*100 if s["n"] else 0
        tss_c = s["tss_cer"]/s["n"]*100 if s["n"] else 0
        print(f"  {p:>5} {s['n']:>5} | {svt_pct:>9.1f}% {svt_c:>7.2f}% | {tss_pct:>9.1f}% {tss_c:>7.2f}%", file=sys.stderr)

    # Show regions where one backend wins and the other loses.
    svt_wins = [r for r in valid if r["svt_exact"] and not r["tss_exact"]]
    tss_wins = [r for r in valid if r["tss_exact"] and not r["svt_exact"]]
    both_wrong = [r for r in valid if not r["svt_exact"] and not r["tss_exact"]]

    print(f"\n  SVTRv2 wins (exact, Tess wrong): {len(svt_wins)}", file=sys.stderr)
    print(f"  Tesseract wins (exact, SVT wrong): {len(tss_wins)}", file=sys.stderr)
    print(f"  Both wrong: {len(both_wrong)}", file=sys.stderr)

    # Show some examples of each winning.
    if tss_wins:
        print(f"\n  Tesseract wins (first 15):", file=sys.stderr)
        for r in tss_wins[:15]:
            print(f"    p{r['page']} VLM: \"{normalize(r['vlm'])[:70]}\"", file=sys.stderr)
            print(f"         SVT: \"{normalize(r['svtrv2'])[:70]}\"", file=sys.stderr)
            print(f"         TSS: \"{normalize(r['tess'])[:70]}\"", file=sys.stderr)

    if svt_wins:
        print(f"\n  SVTRv2 wins (first 15):", file=sys.stderr)
        for r in svt_wins[:15]:
            print(f"    p{r['page']} VLM: \"{normalize(r['vlm'])[:70]}\"", file=sys.stderr)
            print(f"         SVT: \"{normalize(r['svtrv2'])[:70]}\"", file=sys.stderr)
            print(f"         TSS: \"{normalize(r['tess'])[:70]}\"", file=sys.stderr)

    # Save full results.
    out_path = args.output or args.tiff.replace(".tif", "_3way.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": {
            "total": len(results), "valid": n, "errors": len(errors),
            "svt_exact": svt_exact, "tss_exact": tss_exact,
            "svt_space": svt_space, "tss_space": tss_space,
            "svt_cer": round(svt_cer_avg, 4), "tss_cer": round(tss_cer_avg, 4),
        }, "regions": results}, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
