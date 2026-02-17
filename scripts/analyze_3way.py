"""
3-way comparison: C# Tesseract vs Rust SVTRv2 vs VLM (Qwen3 VL).

Uses the VLM comparison results already saved, plus runs Rust recognition
results from the e2e test output, to determine which OCR engine is closest
to VLM "ground truth".

Usage:
    python scripts/analyze_3way.py
"""

import json
import re
import sys
from pathlib import Path

VLM_JSON = Path(r"C:\Temp\vlm_comparison.json")
CSHARP_JSON = Path(r"C:\Temp\tesseract-pipeline\ocr_results.json")


def normalize(s):
    """Normalize for comparison: collapse whitespace, strip."""
    return " ".join(s.split()).strip()


def normalize_numbers(s):
    """Remove spaces within number sequences for fair comparison."""
    # Replace "7 811 518" with "7811518" etc.
    return re.sub(r'(\d)\s+(\d)', r'\1\2', s)


def char_distance(a, b):
    """Simple Levenshtein distance."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    # Use Wagner-Fischer algorithm
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
    """Character Error Rate."""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return char_distance(ref, hyp) / len(ref)


def main():
    if not VLM_JSON.exists():
        print(f"Run vlm_compare.py first to generate {VLM_JSON}")
        sys.exit(1)

    with open(VLM_JSON, encoding="utf-8") as f:
        vlm_results = json.load(f)

    with open(CSHARP_JSON, encoding="utf-8") as f:
        csharp_regions = json.load(f)

    valid = [r for r in vlm_results if r and "error" not in r]
    print(f"VLM results: {len(valid)} valid out of {len(vlm_results)}")

    # Categorize differences.
    categories = {
        "exact_match": 0,
        "number_spacing_only": 0,
        "diacritic_diff": 0,
        "punctuation_only": 0,
        "vlm_hallucination": 0,
        "cs_error": 0,
        "both_differ": 0,
        "minor": 0,
    }

    cs_cer_sum = 0.0
    vlm_cer_count = 0

    diacritic_map = str.maketrans("ÅÄÖåäöØøÆæÐðÞþÜü", "AAOaaoOoAaOoOoUu")

    for r in valid:
        cs = normalize(r["cs_text"])
        vlm = normalize(r["vlm_text"])

        if cs == vlm:
            categories["exact_match"] += 1
            continue

        # Check if only number spacing differs.
        cs_num = normalize_numbers(cs)
        vlm_num = normalize_numbers(vlm)
        if cs_num == vlm_num:
            categories["number_spacing_only"] += 1
            continue

        # Check if only diacritics differ (Å→A, ö→o etc).
        cs_ascii = cs.translate(diacritic_map)
        vlm_ascii = vlm.translate(diacritic_map)
        if normalize_numbers(cs_ascii) == normalize_numbers(vlm_ascii):
            categories["diacritic_diff"] += 1
            continue

        # Check edit distance.
        c = cer(cs, vlm)
        cs_cer_sum += c
        vlm_cer_count += 1

        if c < 0.1:
            categories["minor"] += 1
        elif len(vlm) > len(cs) * 3:
            categories["vlm_hallucination"] += 1
        elif c > 0.8:
            categories["both_differ"] += 1
        else:
            categories["cs_error"] += 1

    print(f"\n=== Difference Categories (VLM vs C#) ===")
    total = len(valid)
    for cat, count in categories.items():
        print(f"  {cat:25s}: {count:4d} ({count/total*100:5.1f}%)")

    truly_same = categories["exact_match"] + categories["number_spacing_only"] + categories["diacritic_diff"]
    print(f"\n  Effectively matching:    {truly_same:4d} ({truly_same/total*100:5.1f}%)")
    print(f"  Truly different:         {total - truly_same:4d} ({(total-truly_same)/total*100:5.1f}%)")

    if vlm_cer_count > 0:
        print(f"\n  Avg CER on differing regions: {cs_cer_sum/vlm_cer_count:.3f}")

    # Show VLM hallucinations.
    print(f"\n=== VLM Hallucinations (VLM text 3x+ longer than C#) ===")
    for r in valid:
        cs = normalize(r["cs_text"])
        vlm = normalize(r["vlm_text"])
        if len(vlm) > len(cs) * 3 and len(cs) > 0:
            print(f"  p{r['page']} row{r['row']}: C#=\"{cs[:50]}\" ({len(cs)} chars)")
            print(f"    VLM=\"{vlm[:80]}\" ({len(vlm)} chars)")

    # Show cases where VLM corrects C# (likely C# errors).
    # These are cases where VLM differs but gives a more plausible reading.
    print(f"\n=== Notable C# vs VLM Differences (likely C# errors) ===")
    corrections = []
    for r in valid:
        cs = normalize(r["cs_text"])
        vlm = normalize(r["vlm_text"])
        if cs == vlm:
            continue

        c = cer(cs, vlm)
        if 0.05 < c < 0.4 and len(cs) > 5:
            corrections.append((r, c))

    corrections.sort(key=lambda x: x[1])
    for r, c in corrections[:30]:
        cs = r["cs_text"][:60]
        vlm = r["vlm_text"][:60]
        print(f"  p{r['page']} row{r['row']} (CER={c:.2f}): C#=\"{cs}\"")
        print(f"    VLM=\"{vlm}\"")


if __name__ == "__main__":
    main()
