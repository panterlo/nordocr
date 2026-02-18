"""
Compare PARSeq and SVTRv2 ONNX models on real text line images.

Usage:
    python compare_onnx.py \
        --parseq-onnx output/parseq_nordic/recognize_parseq.onnx \
        --svtrv2-onnx output/svtrv2_nordic/recognize_svtrv2.onnx \
        --test-dir data/training/splits/test \
        --num-samples 200

    # Single image:
    python compare_onnx.py \
        --parseq-onnx output/parseq_nordic/recognize_parseq.onnx \
        --svtrv2-onnx output/svtrv2_nordic/recognize_svtrv2.onnx \
        --image tests/fixtures/test_line.tiff
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "recognize"))
from charset import NordicTokenizer, NORDIC_CHARSET


def load_onnx_session(onnx_path):
    """Load ONNX model with GPU if available."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()
    print(f"  {Path(onnx_path).name}: {active[0]}")
    return sess


def preprocess_parseq(img, img_height=32, img_width=384):
    """Preprocess for PARSeq: resize + pad to fixed width, normalize."""
    w, h = img.size
    scale = img_height / h
    new_w = min(round(w * scale), img_width)
    img_resized = img.resize((new_w, img_height), Image.BICUBIC)

    # Pad to fixed width with gray
    padded = Image.new("RGB", (img_width, img_height), (128, 128, 128))
    padded.paste(img_resized, (0, 0))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis]  # Add batch dim


def preprocess_svtrv2(img, img_height=32, max_width=768):
    """Preprocess for SVTRv2: resize to height, preserve aspect ratio, cap width."""
    w, h = img.size
    scale = img_height / h
    new_w = max(1, round(w * scale))
    new_w = min(new_w, max_width)
    # Make width divisible by 4 (SVTRv2 stride)
    new_w = max(4, (new_w // 4) * 4)

    img_resized = img.resize((new_w, img_height), Image.BICUBIC)

    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]


def decode_parseq(output, tokenizer):
    """Decode PARSeq output logits."""
    import torch
    probs = torch.from_numpy(output).softmax(-1)
    preds, confs = tokenizer.decode(probs)
    return preds[0]


def decode_ctc(output, charset):
    """Decode CTC output (already softmax from RCTC)."""
    probs = output[0]  # [T, C]
    token_ids = probs.argmax(axis=-1)

    chars = []
    prev = -1
    for t in range(len(token_ids)):
        tok = int(token_ids[t])
        if tok == prev:
            prev = tok
            continue
        prev = tok
        if tok == 0:  # blank
            continue
        if tok < len(charset):
            chars.append(charset[tok])

    return "".join(chars)


def run_single_image(parseq_sess, svtrv2_sess, img, tokenizer, ctc_charset,
                     true_text=None, img_name=""):
    """Run both models on a single image and compare."""
    img_rgb = img.convert("RGB")

    # PARSeq
    inp_p = preprocess_parseq(img_rgb)
    t0 = time.perf_counter()
    out_p = parseq_sess.run(None, {"input": inp_p})[0]
    t_parseq = (time.perf_counter() - t0) * 1000
    pred_parseq = decode_parseq(out_p, tokenizer)

    # SVTRv2
    inp_s = preprocess_svtrv2(img_rgb)
    t0 = time.perf_counter()
    out_s = svtrv2_sess.run(None, {"input": inp_s})[0]
    t_svtrv2 = (time.perf_counter() - t0) * 1000
    pred_svtrv2 = decode_ctc(out_s, ctc_charset)

    result = {
        "img_name": img_name,
        "img_size": img.size,
        "parseq": pred_parseq,
        "svtrv2": pred_svtrv2,
        "parseq_ms": t_parseq,
        "svtrv2_ms": t_svtrv2,
    }
    if true_text is not None:
        result["true"] = true_text
        result["parseq_correct"] = pred_parseq == true_text
        result["svtrv2_correct"] = pred_svtrv2 == true_text

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare PARSeq vs SVTRv2 ONNX")
    parser.add_argument("--parseq-onnx", type=str, required=True)
    parser.add_argument("--svtrv2-onnx", type=str, required=True)
    parser.add_argument("--image", type=str, default=None,
                        help="Single image to test")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Test directory with images/ + labels.tsv")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of test samples to compare")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all predictions, not just disagreements")
    args = parser.parse_args()

    print("Loading ONNX models...")
    parseq_sess = load_onnx_session(args.parseq_onnx)
    svtrv2_sess = load_onnx_session(args.svtrv2_onnx)

    tokenizer = NordicTokenizer()

    # Build CTC charset (matches OpenOCR CTCLabelDecode)
    ctc_charset = ["[blank]"]
    dict_path = Path(__file__).parent.parent / "recognize" / "nordic_dict.txt"
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch:
                ctc_charset.append(ch)

    # Single image mode
    if args.image:
        print(f"\nProcessing: {args.image}")
        img = Image.open(args.image)
        print(f"  Size: {img.size}, Mode: {img.mode}")

        result = run_single_image(parseq_sess, svtrv2_sess, img, tokenizer, ctc_charset,
                                  img_name=Path(args.image).name)
        print(f"\n  PARSeq:  \"{result['parseq']}\"  ({result['parseq_ms']:.1f}ms)")
        print(f"  SVTRv2:  \"{result['svtrv2']}\"  ({result['svtrv2_ms']:.1f}ms)")
        if result["parseq"] == result["svtrv2"]:
            print(f"\n  Models AGREE")
        else:
            print(f"\n  Models DISAGREE")
        return

    # Batch test mode
    if not args.test_dir:
        print("ERROR: Provide --image or --test-dir")
        sys.exit(1)

    test_dir = Path(args.test_dir)
    img_dir = test_dir / "images"
    tsv_path = test_dir / "labels.tsv"

    # Load samples
    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        fn_idx = header.index("filename")
        txt_idx = header.index("text") if "text" in header else header.index("verified_text")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(fn_idx, txt_idx):
                continue
            filename = parts[fn_idx]
            text = parts[txt_idx]
            if text and (img_dir / filename).exists():
                samples.append((filename, text))

    # Sample diverse widths
    import random
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:args.num_samples]

    print(f"\nComparing on {len(samples)} test samples...")
    print(f"{'='*80}")

    parseq_correct = 0
    svtrv2_correct = 0
    both_correct = 0
    disagree_count = 0
    total_parseq_ms = 0
    total_svtrv2_ms = 0

    disagree_examples = []

    for i, (filename, true_text) in enumerate(samples):
        img = Image.open(img_dir / filename)
        result = run_single_image(parseq_sess, svtrv2_sess, img, tokenizer, ctc_charset,
                                  true_text=true_text, img_name=filename)

        total_parseq_ms += result["parseq_ms"]
        total_svtrv2_ms += result["svtrv2_ms"]

        p_ok = result["parseq_correct"]
        s_ok = result["svtrv2_correct"]

        if p_ok:
            parseq_correct += 1
        if s_ok:
            svtrv2_correct += 1
        if p_ok and s_ok:
            both_correct += 1

        if result["parseq"] != result["svtrv2"]:
            disagree_count += 1
            disagree_examples.append(result)

        if args.show_all or result["parseq"] != result["svtrv2"]:
            w = img.size[0]
            marker_p = "OK" if p_ok else "XX"
            marker_s = "OK" if s_ok else "XX"
            if i < 50 or not p_ok or not s_ok:
                print(f"\n[{i+1}] {filename} ({w}px)")
                print(f"  TRUE:    \"{true_text}\"")
                print(f"  PARSeq:  \"{result['parseq']}\" [{marker_p}]")
                print(f"  SVTRv2:  \"{result['svtrv2']}\" [{marker_s}]")

    n = len(samples)
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY ({n} samples)")
    print(f"{'='*80}")
    print(f"  PARSeq accuracy:  {parseq_correct}/{n} ({parseq_correct/n*100:.1f}%)")
    print(f"  SVTRv2 accuracy:  {svtrv2_correct}/{n} ({svtrv2_correct/n*100:.1f}%)")
    print(f"  Both correct:     {both_correct}/{n} ({both_correct/n*100:.1f}%)")
    print(f"  Disagree:         {disagree_count}/{n} ({disagree_count/n*100:.1f}%)")
    print(f"  PARSeq only:      {parseq_correct - both_correct}")
    print(f"  SVTRv2 only:      {svtrv2_correct - both_correct}")
    print(f"")
    print(f"  PARSeq avg time:  {total_parseq_ms/n:.1f}ms/sample")
    print(f"  SVTRv2 avg time:  {total_svtrv2_ms/n:.1f}ms/sample")

    # Show interesting disagreements where one is right and other wrong
    p_only = [r for r in disagree_examples if r["parseq_correct"] and not r["svtrv2_correct"]]
    s_only = [r for r in disagree_examples if r["svtrv2_correct"] and not r["parseq_correct"]]

    if p_only:
        print(f"\nPARSeq correct, SVTRv2 wrong ({len(p_only)} cases):")
        for r in p_only[:10]:
            print(f"  {r['img_name']:30s} TRUE=\"{r['true'][:50]}\" SVTRv2=\"{r['svtrv2'][:50]}\"")

    if s_only:
        print(f"\nSVTRv2 correct, PARSeq wrong ({len(s_only)} cases):")
        for r in s_only[:10]:
            print(f"  {r['img_name']:30s} TRUE=\"{r['true'][:50]}\" PARSeq=\"{r['parseq'][:50]}\"")


if __name__ == "__main__":
    main()
