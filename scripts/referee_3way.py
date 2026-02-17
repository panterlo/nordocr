"""
Cross-reference VLM verdicts with Rust vs C# disagreements.

For each region where Rust and C# text differ, check whether VLM
agrees more with Rust or C#. This tells us who is more accurate.

Reads VLM results from vlm_comparison.json and Rust e2e results.
Since we don't have Rust results in JSON, we hardcode the known
differences from the e2e test output.

Usage:
    python scripts/referee_3way.py
"""

import json
import re
from pathlib import Path


VLM_JSON = Path(r"C:\Temp\vlm_comparison.json")
CSHARP_JSON = Path(r"C:\Temp\tesseract-pipeline\ocr_results.json")


def normalize(s):
    return " ".join(s.split()).strip()


def normalize_numbers(s):
    return re.sub(r'(\d)\s+(\d)', r'\1\2', s)


def edit_dist(a, b):
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


def main():
    with open(VLM_JSON, encoding="utf-8") as f:
        vlm_results = json.load(f)

    with open(CSHARP_JSON, encoding="utf-8") as f:
        csharp_regions = json.load(f)

    # Key insight: VLM results are indexed same as C# regions.
    # For the VLM vs C# analysis, we just need the VLM text.

    # Categorize: for ALL regions, determine VLM agreement.
    # Then summarize what fraction of differences are:
    #   - Number spacing (VLM strips spaces, C# keeps Swedish formatting)
    #   - Diacritics (VLM uses ASCII, C# uses Nordic chars)
    #   - C# errors (VLM has different, likely correct text)
    #   - VLM errors (VLM hallucinated)

    print("=== VLM as Referee: Who's Right? ===\n")

    # Key known differences from our Rust e2e test:
    # Let's categorize what VLM says about the systematic patterns.

    # 1. Page numbers: C# reads "5 (15)", Rust reads "5 (5)"
    #    What does VLM say?
    page_num_cases = []
    for i, vlm_r in enumerate(vlm_results):
        if vlm_r is None or "error" in vlm_r:
            continue
        cs = csharp_regions[i]
        cs_text = cs["text"]
        # Look for "N (15)" pattern
        if re.match(r'^\d+\s*\(\d+\)$', cs_text.strip()):
            vlm_text = vlm_r["vlm_text"]
            page_num_cases.append((cs["pageNumber"], cs_text, vlm_text))

    if page_num_cases:
        print("1. PAGE NUMBER HEADERS (e.g. '5 (15)'):")
        print(f"   Rust reads these as 'N (5)' — dropping the '1' in '15'")
        for p, cs, vlm in page_num_cases:
            print(f"   p{p}: C#=\"{cs}\" VLM=\"{vlm}\"")
        print()

    # 2. Number spacing: C# "7811 518" vs Rust "7 811 518"
    num_spacing = {"cs_style": 0, "rust_style": 0, "vlm_no_spaces": 0, "other": 0}
    for i, vlm_r in enumerate(vlm_results):
        if vlm_r is None or "error" in vlm_r:
            continue
        cs_text = normalize(csharp_regions[i]["text"])
        vlm_text = normalize(vlm_r["vlm_text"])
        # Only look at pure number regions
        if not re.match(r'^-?[\d\s]+$', cs_text):
            continue
        if cs_text == vlm_text:
            num_spacing["cs_style"] += 1
        elif normalize_numbers(cs_text) == normalize_numbers(vlm_text):
            num_spacing["vlm_no_spaces"] += 1
        else:
            num_spacing["other"] += 1

    print("2. NUMBER SPACING (pure numeric regions):")
    print(f"   VLM matches C# spacing:     {num_spacing['cs_style']}")
    print(f"   VLM strips spaces:          {num_spacing['vlm_no_spaces']}")
    print(f"   Other differences:          {num_spacing['other']}")
    print()

    # 3. Specific C# errors that VLM corrects:
    print("3. C# ERRORS CONFIRMED BY VLM:")
    cs_errors = []
    for i, vlm_r in enumerate(vlm_results):
        if vlm_r is None or "error" in vlm_r:
            continue
        cs = csharp_regions[i]
        cs_text = normalize(cs["text"])
        vlm_text = normalize(vlm_r["vlm_text"])
        if cs_text == vlm_text:
            continue
        # Known C# errors: "O" instead of "0", wrong dates, typos
        if "O84" in cs_text and "084" in vlm_text:
            cs_errors.append((cs["pageNumber"], cs_text, vlm_text, "O vs 0"))
        elif "5356300" in cs_text and "556500" in vlm_text:
            cs_errors.append((cs["pageNumber"], cs_text, vlm_text, "wrong org.nr"))
        elif "Reseryv" in cs_text and "Reserv" in vlm_text:
            cs_errors.append((cs["pageNumber"], cs_text, vlm_text, "typo"))
        elif "Icasing" in cs_text and "leasing" in vlm_text.lower():
            cs_errors.append((cs["pageNumber"], cs_text, vlm_text, "I vs l"))
        elif "f2 14" in cs_text or "12-75" in cs_text or "12- /4" in cs_text:
            cs_errors.append((cs["pageNumber"], cs_text, vlm_text, "wrong date"))

    for p, cs, vlm, reason in cs_errors:
        print(f"   p{p}: C#=\"{cs}\" VLM=\"{vlm}\" [{reason}]")
    print()

    # 4. Overall CER comparison: C# vs VLM
    print("4. CHARACTER ERROR RATE (CER) — C# vs VLM reference:")
    print("   (Using VLM as ground truth, ignoring number spacing)")
    total_chars = 0
    total_errors = 0
    for i, vlm_r in enumerate(vlm_results):
        if vlm_r is None or "error" in vlm_r:
            continue
        cs_text = normalize(csharp_regions[i]["text"])
        vlm_text = normalize(vlm_r["vlm_text"])

        # Skip VLM hallucinations (3x+ length)
        if len(vlm_text) > len(cs_text) * 3 and len(cs_text) > 3:
            continue

        # Normalize number spacing for fair comparison
        cs_n = normalize_numbers(cs_text)
        vlm_n = normalize_numbers(vlm_text)

        ref_len = max(len(vlm_n), 1)
        errors = edit_dist(cs_n, vlm_n)
        total_chars += ref_len
        total_errors += errors

    print(f"   Total reference chars: {total_chars}")
    print(f"   Total edit errors:     {total_errors}")
    print(f"   Overall CER:           {total_errors/total_chars*100:.2f}%")


if __name__ == "__main__":
    main()
