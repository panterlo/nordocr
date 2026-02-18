"""
Evaluate a trained Nordic OCR recognition model.

Supports both PARSeq and SVTRv2 (CTC) models with the full Nordic metrics suite:
CER, WER, exact-match accuracy, diacritical accuracy, per-character diacritical
breakdown, and character confusion matrix.

Usage:
    # PARSeq
    python evaluate.py \
        --model parseq \
        --checkpoint output/parseq_nordic/best.pth \
        --test-dir D:/TrainingData/splits/test \
        --output-dir output/parseq_nordic/eval

    # SVTRv2 (CTC)
    python evaluate.py \
        --model svtrv2 \
        --checkpoint output/svtrv2_nordic/best.pth \
        --test-dir D:/TrainingData/splits/test \
        --output-dir output/svtrv2_nordic/eval

    # SVTRv2 with explicit config and OpenOCR path
    python evaluate.py \
        --model svtrv2 \
        --checkpoint output/svtrv2_nordic/best.pth \
        --config output/svtrv2_nordic/config.yml \
        --openocr-dir C:/dev/OpenOCR \
        --test-dir D:/TrainingData/splits/test

    # Compare two models on the same test set
    python evaluate.py --model parseq --checkpoint A/best.pth --test-dir test/ --output-dir eval_parseq
    python evaluate.py --model svtrv2 --checkpoint B/best.pth --test-dir test/ --output-dir eval_svtrv2
    # Then compare eval_parseq/metrics.json vs eval_svtrv2/metrics.json
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "recognize"))
from charset import NordicTokenizer, NORDIC_CHARSET, DIACRITICAL_CHARS, CONFUSION_PAIRS


# ---------------------------------------------------------------------------
# Metrics (shared by both models)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1, s2):
    """Compute Levenshtein edit distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def compute_metrics(all_predictions):
    """Compute full Nordic metrics from a list of (pred_text, true_text) pairs."""
    confusion_counts = Counter()
    diac_errors = Counter()
    results = []

    total_chars = 0
    total_char_errors = 0
    total_words = 0
    total_word_errors = 0
    total_correct = 0
    total_samples = 0
    diac_total = 0
    diac_correct = 0

    for pred_text, true_text in all_predictions:
        # Character Error Rate
        char_errors = levenshtein_distance(pred_text, true_text)
        total_char_errors += char_errors
        total_chars += len(true_text)

        # Word Error Rate
        pred_words = pred_text.split()
        true_words = true_text.split()
        word_errors = levenshtein_distance(pred_words, true_words)
        total_word_errors += word_errors
        total_words += len(true_words)

        # Exact match
        total_samples += 1
        if pred_text == true_text:
            total_correct += 1

        # Character-level confusion tracking (alignment by position)
        min_len = min(len(pred_text), len(true_text))
        for j in range(min_len):
            if true_text[j] != pred_text[j]:
                confusion_counts[(true_text[j], pred_text[j])] += 1

            # Diacritical accuracy
            if true_text[j] in DIACRITICAL_CHARS:
                diac_total += 1
                if pred_text[j] == true_text[j]:
                    diac_correct += 1
                else:
                    diac_errors[(true_text[j], pred_text[j])] += 1

        # Count diacriticals in positions beyond pred length as errors
        for j in range(min_len, len(true_text)):
            if true_text[j] in DIACRITICAL_CHARS:
                diac_total += 1
                diac_errors[(true_text[j], "<missing>")] += 1

        results.append({
            "true": true_text,
            "pred": pred_text,
            "cer": char_errors / max(len(true_text), 1),
            "correct": pred_text == true_text,
        })

    metrics = {
        "cer": total_char_errors / max(total_chars, 1),
        "wer": total_word_errors / max(total_words, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "diacritical_accuracy": diac_correct / max(diac_total, 1),
        "num_samples": total_samples,
        "num_chars": total_chars,
        "num_diacritical_chars": diac_total,
    }

    # Top confusion pairs
    top_confusions = confusion_counts.most_common(20)
    metrics["top_confusions"] = [
        {"true": t, "pred": p, "count": c} for (t, p), c in top_confusions
    ]

    # Diacritical confusion matrix
    metrics["diacritical_errors"] = [
        {"true": t, "pred": p, "count": c}
        for (t, p), c in diac_errors.most_common(20)
    ]

    # Per-diacritical accuracy
    diac_per_char = {}
    for c in sorted(DIACRITICAL_CHARS):
        total_c = sum(1 for r in results for ch in r["true"] if ch == c)
        correct_c = total_c - sum(
            v for (t, _), v in diac_errors.items() if t == c
        )
        if total_c > 0:
            diac_per_char[c] = {
                "total": total_c,
                "correct": correct_c,
                "accuracy": correct_c / total_c,
            }
    metrics["per_diacritical"] = diac_per_char

    # Width-bucketed accuracy (for SVTRv2 analysis — how well do wide lines do?)
    width_buckets = {}
    for r in results:
        text_len = len(r["true"])
        if text_len <= 10:
            bucket = "short (1-10)"
        elif text_len <= 30:
            bucket = "medium (11-30)"
        elif text_len <= 60:
            bucket = "long (31-60)"
        else:
            bucket = "very long (61+)"
        if bucket not in width_buckets:
            width_buckets[bucket] = {"total": 0, "correct": 0, "total_cer": 0.0}
        width_buckets[bucket]["total"] += 1
        width_buckets[bucket]["correct"] += int(r["correct"])
        width_buckets[bucket]["total_cer"] += r["cer"]
    for bucket, info in width_buckets.items():
        info["accuracy"] = info["correct"] / max(info["total"], 1)
        info["avg_cer"] = info["total_cer"] / max(info["total"], 1)
        del info["total_cer"]
    metrics["by_length"] = width_buckets

    return metrics, results


# ---------------------------------------------------------------------------
# Test dataset (TSV + images, variable-width for SVTRv2)
# ---------------------------------------------------------------------------

class TestDataset(Dataset):
    """Test dataset that loads images at their natural aspect ratio.

    For PARSeq: resizes to fixed (H, max_W) with aspect-preserving padding.
    For SVTRv2: resizes to height H, preserves natural width (no max_W cap).
    """

    def __init__(self, data_dir, img_height=32, max_width=None, max_width_cap=None, max_label_len=100):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.img_height = img_height
        self.max_width = max_width  # None = variable width (SVTRv2)
        self.max_width_cap = max_width_cap  # Cap width without padding (SVTRv2)
        self.charset_set = set(NORDIC_CHARSET)

        # Load labels
        self.samples = []
        tsv_path = self.data_dir / "labels.tsv"
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < len(header):
                    continue
                row = dict(zip(header, fields))
                filename = row["filename"]
                text = row.get("text") or row.get("verified_text") or ""
                # Keep only charset-valid characters
                filtered = "".join(c for c in text if c in self.charset_set)
                if filtered and len(filtered) <= max_label_len:
                    img_path = self.img_dir / filename
                    if img_path.exists():
                        self.samples.append({"filename": filename, "text": filtered})

        # Shared transforms: ToTensor + Normalize (matching inference kernel)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(self.img_dir / sample["filename"]).convert("RGB")
        w, h = img.size

        # Resize to target height, preserving aspect ratio
        scale = self.img_height / h
        new_w = max(1, round(w * scale))

        if self.max_width is not None:
            # PARSeq mode: cap width, pad to max_width
            new_w = min(new_w, self.max_width)
            img = img.resize((new_w, self.img_height), Image.BICUBIC)
            # Pad to fixed max_width with gray (128 → 0.0 after normalize)
            padded = Image.new("RGB", (self.max_width, self.img_height), (128, 128, 128))
            padded.paste(img, (0, 0))
            img = padded
        else:
            # SVTRv2 mode: variable width, cap if needed (done in collate)
            if self.max_width_cap is not None:
                new_w = min(new_w, self.max_width_cap)
            img = img.resize((new_w, self.img_height), Image.BICUBIC)

        img = self.to_tensor(img)
        return img, sample["text"], new_w


def svtrv2_collate_fn(batch):
    """Collate for SVTRv2: pad images to the max width in the batch.

    Padding value is 0.0 (gray after normalization), matching inference behavior.
    """
    images, labels, widths = zip(*batch)
    max_w = max(img.shape[2] for img in images)

    padded = []
    for img in images:
        # img is [3, H, W] — pad W dimension to max_w
        pad_w = max_w - img.shape[2]
        if pad_w > 0:
            # F.pad order: (left, right, top, bottom) for 3D tensor
            img = torch.nn.functional.pad(img, (0, pad_w), value=0.0)
        padded.append(img)

    return torch.stack(padded, 0), list(labels), list(widths)


def parseq_collate_fn(batch):
    """Collate for PARSeq: images are already fixed-width from dataset."""
    images, labels, widths = zip(*batch)
    return torch.stack(images, 0), list(labels), list(widths)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_parseq(checkpoint_path, device, max_label_len):
    """Load PARSeq model from a training checkpoint."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "recognize"))
    from train_parseq import load_parseq_model

    tokenizer = NordicTokenizer()
    model = load_parseq_model(tokenizer, pretrained_name="parseq", max_label_length=max_label_len)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]
    # Strip _orig_mod. prefix from torch.compile'd checkpoints
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if "metrics" in checkpoint:
        print(f"  Checkpoint metrics: {checkpoint['metrics']}")

    return model, tokenizer


def find_openocr(openocr_dir=None):
    """Find OpenOCR installation."""
    if openocr_dir:
        p = Path(openocr_dir)
        if (p / "openrec" / "modeling").exists():
            return p

    for candidate in [
        Path("C:/dev/OpenOCR"),
        Path.home() / "OpenOCR",
        Path(__file__).parent.parent.parent / "OpenOCR",
        Path("OpenOCR"),
    ]:
        if (candidate / "openrec" / "modeling").exists():
            return candidate
    return None


def load_svtrv2(checkpoint_path, config_path, openocr_dir, device):
    """Load SVTRv2 model from an OpenOCR training checkpoint."""
    openocr_path = find_openocr(openocr_dir)
    if openocr_path is None:
        print("ERROR: OpenOCR not found. Pass --openocr-dir or clone to C:/dev/OpenOCR")
        sys.exit(1)
    print(f"  OpenOCR: {openocr_path}")

    sys.path.insert(0, str(openocr_path))
    from openrec.modeling import build_model
    from openrec.postprocess import build_post_process
    from tools.engine.config import Config

    # Find config
    if config_path is None:
        checkpoint_dir = Path(checkpoint_path).parent
        for candidate in [
            checkpoint_dir / "config.yml",
            checkpoint_dir / "config.yaml",
            checkpoint_dir.parent / "config.yml",
        ]:
            if candidate.exists():
                config_path = str(candidate)
                break
        if config_path is None:
            print(f"ERROR: No config.yml found near {checkpoint_path}. Pass --config.")
            sys.exit(1)
    print(f"  Config: {config_path}")

    cfg = Config(config_path)
    _cfg = cfg.cfg

    # Build post-processor to get charset size and CTC decoder
    post_process = build_post_process(_cfg["PostProcess"])
    char_num = len(getattr(post_process, "character"))
    _cfg["Architecture"]["Decoder"]["out_channels"] = char_num

    # Build and load model
    model = build_model(_cfg["Architecture"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    # Rep reparameterization
    for layer in model.modules():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped", True):
            layer.rep()

    model = model.to(device)
    model.eval()

    # Build the CTC charset mapping (blank at index 0, then dict chars)
    # This matches OpenOCR's CTCLabelDecode: character = ['blank'] + dict_chars
    character = getattr(post_process, "character")

    print(f"  Architecture: {_cfg['Architecture']['algorithm']}")
    print(f"  Decoder: {_cfg['Architecture']['Decoder']['name']}")
    print(f"  Output classes: {char_num} (blank + {char_num - 1} charset chars)")

    return model, character, post_process


# ---------------------------------------------------------------------------
# CTC decoding (Python, matching Rust decode_ctc)
# ---------------------------------------------------------------------------

def ctc_decode_batch(logits, character, is_softmax=False):
    """Decode CTC logits/probs to text strings.

    Args:
        logits: [B, T, C] numpy array or tensor of raw logits or softmax probs.
        character: list of characters where index 0 = blank.
        is_softmax: if True, input is already softmax probabilities (e.g. RCTC decoder).

    Returns:
        list of (text, confidence) tuples.
    """
    import numpy as np
    if torch.is_tensor(logits):
        logits = logits.cpu().float().numpy()

    batch_size = logits.shape[0]
    results = []

    for b in range(batch_size):
        sample_logits = logits[b]  # [T, C]

        if is_softmax:
            probs = sample_logits
        else:
            # Softmax
            exp_logits = np.exp(sample_logits - sample_logits.max(axis=-1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Argmax per timestep
        token_ids = probs.argmax(axis=-1)  # [T]
        token_probs = probs[np.arange(len(token_ids)), token_ids]  # [T]

        # CTC collapse: remove consecutive duplicates, then remove blanks
        text_chars = []
        char_probs = []
        prev_token = -1

        for t in range(len(token_ids)):
            token = int(token_ids[t])
            if token == prev_token:
                prev_token = token
                continue
            prev_token = token

            # Skip blank (index 0)
            if token == 0:
                continue

            # Map token to character
            if token < len(character):
                text_chars.append(character[token])
                char_probs.append(float(token_probs[t]))

        text = "".join(text_chars)

        # Geometric mean confidence
        if char_probs:
            log_prob_sum = sum(np.log(max(p, 1e-10)) for p in char_probs)
            confidence = float(np.exp(log_prob_sum / len(char_probs)))
        else:
            confidence = 0.0

        results.append((text, confidence))

    return results


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def run_parseq_inference(model, tokenizer, dataloader, device):
    """Run PARSeq inference and return (pred_text, true_text) pairs."""
    predictions = []
    model.eval()

    with torch.no_grad():
        for images, labels, widths in tqdm(dataloader, desc="PARSeq inference"):
            images = images.to(device)
            logits = model.forward(tokenizer, images)
            probs = logits.softmax(-1)
            preds, _ = tokenizer.decode(probs)

            for pred_text, true_text in zip(preds, labels):
                predictions.append((pred_text, true_text))

    return predictions


def run_svtrv2_inference(model, character, dataloader, device):
    """Run SVTRv2 CTC inference and return (pred_text, true_text) pairs."""
    predictions = []
    model.eval()

    with torch.no_grad():
        for images, labels, widths in tqdm(dataloader, desc="SVTRv2 inference"):
            images = images.to(device)
            output = model(images)
            # SVTRv2 may return (features, logits) tuple
            if isinstance(output, (tuple, list)):
                logits = output[-1]
            else:
                logits = output

            # RCTC decoder applies softmax in eval mode
            decoded = ctc_decode_batch(logits, character, is_softmax=True)

            for (pred_text, _confidence), true_text in zip(decoded, labels):
                predictions.append((pred_text, true_text))

    return predictions


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_metrics(metrics, model_name):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {model_name}")
    print(f"{'='*60}")
    print(f"  Samples:              {metrics['num_samples']}")
    print(f"  CER:                  {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
    print(f"  WER:                  {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
    print(f"  Exact match accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Diacritical accuracy: {metrics['diacritical_accuracy']:.4f} ({metrics['diacritical_accuracy']*100:.2f}%)")
    print(f"  Diacritical chars:    {metrics['num_diacritical_chars']}")

    if metrics.get("per_diacritical"):
        print(f"\n  Per-character diacritical accuracy:")
        for char, info in sorted(metrics["per_diacritical"].items()):
            print(f"    {char}: {info['accuracy']:.4f} ({info['correct']}/{info['total']})")

    if metrics.get("by_length"):
        print(f"\n  Accuracy by text length:")
        for bucket in ["short (1-10)", "medium (11-30)", "long (31-60)", "very long (61+)"]:
            if bucket in metrics["by_length"]:
                info = metrics["by_length"][bucket]
                print(f"    {bucket:20s}: acc={info['accuracy']:.4f}  "
                      f"avg_cer={info['avg_cer']:.4f}  (n={info['total']})")

    if metrics.get("top_confusions"):
        print(f"\n  Top character confusions:")
        for item in metrics["top_confusions"][:10]:
            print(f"    '{item['true']}' -> '{item['pred']}': {item['count']} times")

    if metrics.get("diacritical_errors"):
        print(f"\n  Diacritical errors:")
        for item in metrics["diacritical_errors"][:10]:
            print(f"    '{item['true']}' -> '{item['pred']}': {item['count']} times")


def save_results(metrics, results, output_dir):
    """Save metrics JSON and predictions TSV."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "predictions.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true", "pred", "cer", "correct"], delimiter="\t")
        writer.writeheader()
        writer.writerows(results)

    # Save worst predictions for manual review
    worst = sorted(results, key=lambda r: -r["cer"])[:50]
    with open(os.path.join(output_dir, "worst_50.tsv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true", "pred", "cer", "correct"], delimiter="\t")
        writer.writeheader()
        writer.writerows(worst)

    print(f"\nResults saved to: {output_dir}")
    print(f"  metrics.json     — all metrics")
    print(f"  predictions.tsv  — every prediction")
    print(f"  worst_50.tsv     — 50 worst predictions for review")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Nordic OCR model (PARSeq or SVTRv2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, choices=["parseq", "svtrv2"],
                        help="Model type: 'parseq' or 'svtrv2'")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Test directory (images/ + labels.tsv)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: checkpoint dir)")
    parser.add_argument("--config", type=str, default=None,
                        help="OpenOCR config YAML (SVTRv2 only; auto-detected from checkpoint dir)")
    parser.add_argument("--openocr-dir", type=str, default=None,
                        help="Path to OpenOCR repo (SVTRv2 only; auto-detected)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-label-len", type=int, default=100,
                        help="Max label length (100 for full-width lines)")
    parser.add_argument("--img-height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=None,
                        help="Fixed image width (PARSeq default: 384, SVTRv2: variable)")
    parser.add_argument("--max-width-cap", type=int, default=None,
                        help="Cap image width without padding (SVTRv2: match training max_ratio*32)")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Set sensible defaults per model
    if args.max_width is None:
        if args.model == "parseq":
            args.max_width = 384
        # SVTRv2: None = variable width
    if args.max_width_cap is None and args.model == "svtrv2":
        # Default cap to 768px (max_ratio=24 * height=32) matching typical training
        args.max_width_cap = 768

    # Load model
    print(f"\nLoading {args.model} model: {args.checkpoint}")
    if args.model == "parseq":
        model, tokenizer = load_parseq(args.checkpoint, device, args.max_label_len)
    else:
        model, character, post_process = load_svtrv2(
            args.checkpoint, args.config, args.openocr_dir, device
        )

    # Load test dataset
    print(f"\nLoading test set: {args.test_dir}")
    test_dataset = TestDataset(
        args.test_dir,
        img_height=args.img_height,
        max_width=args.max_width,
        max_width_cap=getattr(args, 'max_width_cap', None),
        max_label_len=args.max_label_len,
    )
    print(f"  Test samples: {len(test_dataset)}")

    collate = parseq_collate_fn if args.model == "parseq" else svtrv2_collate_fn
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # Run inference
    print(f"\nRunning inference...")
    if args.model == "parseq":
        predictions = run_parseq_inference(model, tokenizer, test_loader, device)
    else:
        predictions = run_svtrv2_inference(model, character, test_loader, device)

    # Compute metrics
    metrics, results = compute_metrics(predictions)
    metrics["model"] = args.model
    metrics["checkpoint"] = args.checkpoint

    # Print
    model_label = f"PARSeq ({args.max_width}px)" if args.model == "parseq" else "SVTRv2 CTC (variable-width)"
    print_metrics(metrics, model_label)

    # Save
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    save_results(metrics, results, output_dir)


if __name__ == "__main__":
    main()
