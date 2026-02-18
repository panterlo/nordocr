"""
End-to-end OCR test on a real TIFF page using ONNX models.

Extracts text lines via horizontal projection, then runs both
PARSeq and SVTRv2 ONNX recognition models and compares results.

Usage:
    python test_page_onnx.py \
        --tiff /home/sysop/06313923_001_20260209T120358.tif \
        --page 4 \
        --parseq-onnx output/parseq_nordic/recognize_parseq.onnx \
        --svtrv2-onnx output/svtrv2_nordic/recognize_svtrv2.onnx
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "recognize"))
from charset import NordicTokenizer


def load_onnx_session(onnx_path, use_gpu=True):
    """Load ONNX model, prefer GPU."""
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()[0]
    return sess, active


def extract_text_lines(page_gray, min_line_height=15, margin=4):
    """Extract text line images from a grayscale page using horizontal projection."""
    binary = (page_gray < 160).astype(np.uint8)
    h_proj = binary.sum(axis=1)

    # Find text line regions
    threshold = max(5, h_proj.max() * 0.005)
    in_line = h_proj > threshold
    transitions = np.diff(in_line.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1

    # Pair starts with ends
    if len(ends) > 0 and (len(starts) == 0 or ends[0] < starts[0]):
        ends = ends[1:]
    n = min(len(starts), len(ends))
    starts = starts[:n]
    ends = ends[:n]

    # Merge close lines and filter noise
    lines = []
    i = 0
    while i < n:
        y0 = max(0, starts[i] - margin)
        y1 = min(page_gray.shape[0], ends[i] + margin)

        # Merge with next line if gap is small
        while i + 1 < n and starts[i + 1] - ends[i] < min_line_height:
            i += 1
            y1 = min(page_gray.shape[0], ends[i] + margin)

        h = y1 - y0
        if h >= min_line_height:
            # Crop line, trim horizontal whitespace
            line_img = page_gray[y0:y1, :]
            col_proj = (line_img < 160).sum(axis=0)
            cols = np.where(col_proj > 0)[0]
            if len(cols) > 0:
                x0 = max(0, cols[0] - margin)
                x1 = min(page_gray.shape[1], cols[-1] + margin)
                line_crop = page_gray[y0:y1, x0:x1]
                lines.append({
                    "image": line_crop,
                    "bbox": (x0, y0, x1, y1),
                    "width": x1 - x0,
                    "height": h,
                })
        i += 1

    return lines


def preprocess_parseq(img_gray, img_height=32, img_width=384):
    """Preprocess for PARSeq: grayscale->RGB, resize+pad to fixed width."""
    h, w = img_gray.shape
    img = Image.fromarray(img_gray).convert("RGB")

    scale = img_height / h
    new_w = min(round(w * scale), img_width)
    img_resized = img.resize((new_w, img_height), Image.BICUBIC)

    padded = Image.new("RGB", (img_width, img_height), (128, 128, 128))
    padded.paste(img_resized, (0, 0))

    arr = np.array(padded, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]


def preprocess_svtrv2(img_gray, img_height=32, max_width=768):
    """Preprocess for SVTRv2: grayscale->RGB, resize to height, variable width."""
    h, w = img_gray.shape
    img = Image.fromarray(img_gray).convert("RGB")

    scale = img_height / h
    new_w = max(1, round(w * scale))
    new_w = min(new_w, max_width)
    new_w = max(4, (new_w // 4) * 4)  # Align to stride 4

    img_resized = img.resize((new_w, img_height), Image.BICUBIC)

    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]


def decode_parseq(output, tokenizer):
    """Decode PARSeq output logits."""
    import torch
    probs = torch.from_numpy(output).softmax(-1)
    preds, _ = tokenizer.decode(probs)
    return preds[0]


def decode_ctc(output, charset):
    """Decode CTC output (already softmax from RCTC in eval mode)."""
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
        if tok == 0:
            continue
        if tok < len(charset):
            chars.append(charset[tok])
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser(description="End-to-end TIFF OCR test")
    parser.add_argument("--tiff", type=str, required=True)
    parser.add_argument("--page", type=int, default=0, help="0-indexed page number")
    parser.add_argument("--parseq-onnx", type=str, required=True)
    parser.add_argument("--svtrv2-onnx", type=str, required=True)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--save-lines", type=str, default=None,
                        help="Save extracted line images to this directory")
    args = parser.parse_args()

    # Load TIFF page
    print(f"Loading TIFF: {args.tiff}")
    img = Image.open(args.tiff)
    print(f"  Total pages: {img.n_frames}")
    img.seek(args.page)
    page_gray = np.array(img.convert("L"))
    print(f"  Page {args.page + 1}: {page_gray.shape[1]}x{page_gray.shape[0]}")

    # Extract text lines
    t0 = time.perf_counter()
    lines = extract_text_lines(page_gray)
    t_detect = (time.perf_counter() - t0) * 1000
    print(f"  Detected {len(lines)} text lines ({t_detect:.1f}ms)")

    if args.save_lines:
        import os
        os.makedirs(args.save_lines, exist_ok=True)
        for i, line in enumerate(lines):
            Image.fromarray(line["image"]).save(f"{args.save_lines}/line_{i:03d}.png")
        print(f"  Saved line images to {args.save_lines}/")

    if not lines:
        print("No text lines found!")
        return

    # Load ONNX models
    print(f"\nLoading models...")
    parseq_sess, parseq_ep = load_onnx_session(args.parseq_onnx, args.gpu)
    svtrv2_sess, svtrv2_ep = load_onnx_session(args.svtrv2_onnx, args.gpu)
    print(f"  PARSeq: {parseq_ep}")
    print(f"  SVTRv2: {svtrv2_ep}")

    tokenizer = NordicTokenizer()

    # CTC charset
    ctc_charset = ["[blank]"]
    dict_path = Path(__file__).parent.parent / "recognize" / "nordic_dict.txt"
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n")
            if ch:
                ctc_charset.append(ch)

    # Warmup
    dummy_p = preprocess_parseq(lines[0]["image"])
    dummy_s = preprocess_svtrv2(lines[0]["image"])
    parseq_sess.run(None, {"input": dummy_p})
    svtrv2_sess.run(None, {"input": dummy_s})

    # Run both models on each line
    print(f"\n{'='*100}")
    print(f"PAGE {args.page + 1} — RECOGNITION RESULTS")
    print(f"{'='*100}")

    total_parseq_ms = 0
    total_svtrv2_ms = 0
    agree_count = 0
    disagree_lines = []

    for i, line in enumerate(lines):
        img_gray = line["image"]
        w = line["width"]

        # PARSeq
        inp_p = preprocess_parseq(img_gray)
        t0 = time.perf_counter()
        out_p = parseq_sess.run(None, {"input": inp_p})[0]
        t_p = (time.perf_counter() - t0) * 1000
        pred_p = decode_parseq(out_p, tokenizer)
        total_parseq_ms += t_p

        # SVTRv2
        inp_s = preprocess_svtrv2(img_gray)
        t0 = time.perf_counter()
        out_s = svtrv2_sess.run(None, {"input": inp_s})[0]
        t_s = (time.perf_counter() - t0) * 1000
        pred_s = decode_ctc(out_s, ctc_charset)
        total_svtrv2_ms += t_s

        agree = pred_p == pred_s
        if agree:
            agree_count += 1
            marker = ""
        else:
            marker = " <-- DIFFER"
            disagree_lines.append(i)

        print(f"\nLine {i+1:2d} ({w}px):{marker}")
        if agree:
            print(f"  BOTH:    \"{pred_p}\"")
        else:
            print(f"  PARSeq:  \"{pred_p}\"")
            print(f"  SVTRv2:  \"{pred_s}\"")

    n = len(lines)
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"  Lines:              {n}")
    print(f"  Models agree:       {agree_count}/{n} ({agree_count/n*100:.1f}%)")
    print(f"  Models disagree:    {n - agree_count}/{n}")
    print(f"  PARSeq total:       {total_parseq_ms:.1f}ms ({total_parseq_ms/n:.1f}ms/line)")
    print(f"  SVTRv2 total:       {total_svtrv2_ms:.1f}ms ({total_svtrv2_ms/n:.1f}ms/line)")
    print(f"  PARSeq throughput:  {n/(total_parseq_ms/1000):.0f} lines/sec")
    print(f"  SVTRv2 throughput:  {n/(total_svtrv2_ms/1000):.0f} lines/sec")

    # Print full page text from each model
    print(f"\n{'='*100}")
    print(f"FULL PAGE TEXT — SVTRv2")
    print(f"{'='*100}")
    for i, line in enumerate(lines):
        inp_s = preprocess_svtrv2(line["image"])
        out_s = svtrv2_sess.run(None, {"input": inp_s})[0]
        print(decode_ctc(out_s, ctc_charset))


if __name__ == "__main__":
    main()
