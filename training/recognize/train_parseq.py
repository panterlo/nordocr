"""
PARSeq fine-tuning for Nordic text recognition.

Standalone training script that:
1. Downloads pretrained PARSeq weights from GitHub releases
2. Builds the model with Nordic 128-token vocab (125 chars + 3 specials)
3. Transfers compatible pretrained weights (encoder + decoder structure)
4. Reinitializes output head and token embeddings for new charset
5. Trains with PARSeq's permutation-aware loss
6. Evaluates diacritical accuracy throughout training

Prerequisites:
    pip install -e path/to/parseq  # clone https://github.com/baudm/parseq
    # OR the script falls back to torch.hub (auto-downloads repo)

Usage:
    # Single GPU
    python train_parseq.py \
        --auto-labeled /home/sysop/nordocr/data/training/auto_labeled \
        --spot-check /home/sysop/nordocr/data/spot_check \
        --output-dir output/parseq_nordic

    # Multi-GPU (2x RTX 6000 PRO Blackwell)
    torchrun --nproc_per_node=2 train_parseq.py \
        --auto-labeled /home/sysop/nordocr/data/training/auto_labeled \
        --spot-check /home/sysop/nordocr/data/spot_check \
        --output-dir output/parseq_nordic
"""

import argparse
import math
import os
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from charset import NordicTokenizer, NORDIC_CHARSET, DIACRITICAL_CHARS
from dataset import build_train_dataset, build_val_dataset, collate_fn


# ---------------------------------------------------------------------------
# PARSeq model loading
# ---------------------------------------------------------------------------

PRETRAINED_URLS = {
    "parseq": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq-bb5792a6.pt",
    "parseq-tiny": "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt",
}

# Default PARSeq hyperparameters
PARSEQ_CONFIG = {
    "img_size": (32, 384),
    "patch_size": (4, 8),
    "embed_dim": 384,
    "enc_num_heads": 6,
    "enc_mlp_ratio": 4,
    "enc_depth": 12,
    "dec_num_heads": 12,
    "dec_mlp_ratio": 4,
    "dec_depth": 1,
    "dropout": 0.1,
    "decode_ar": True,
    "refine_iters": 1,
}


def load_parseq_model(tokenizer, pretrained_name="parseq", max_label_length=25):
    """
    Build PARSeq model with Nordic charset and load pretrained encoder weights.

    Tries to import from installed strhub package first, falls back to torch.hub.
    """
    num_tokens = tokenizer.vocab_size  # 128 for Nordic (125 + 3)

    # Try to load PARSeq model class
    try:
        from strhub.models.parseq.model import PARSeq
        print("Using locally installed strhub package")
    except ImportError:
        print("strhub not installed, loading via torch.hub...")
        # torch.hub will download the parseq repo
        hub_model = torch.hub.load("baudm/parseq", pretrained_name, pretrained=False, trust_repo=True)
        # Get the inner model class
        PARSeq = type(hub_model.model if hasattr(hub_model, "model") else hub_model)

    # Build model with our vocab size
    cfg = PARSEQ_CONFIG.copy()
    model = PARSeq(
        num_tokens=num_tokens,
        max_label_length=max_label_length,
        **cfg,
    )

    # Load pretrained weights (partial transfer)
    url = PRETRAINED_URLS.get(pretrained_name)
    if url:
        print(f"Downloading pretrained weights: {pretrained_name}")
        pretrained_sd = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)

        model_sd = model.state_dict()
        transferred = {}
        skipped = []

        for k, v in pretrained_sd.items():
            if k in model_sd and v.shape == model_sd[k].shape:
                transferred[k] = v
            else:
                skipped.append(k)

        model.load_state_dict(transferred, strict=False)
        print(f"  Transferred {len(transferred)}/{len(model_sd)} parameter tensors")
        if skipped:
            print(f"  Skipped (shape mismatch): {skipped}")

        # Partial transfer of token embeddings — copy common chars
        # Pretrained charset: 94 chars -> tokens [EOS, c1..c94, BOS, PAD] = 97 total
        # Our charset: 125 chars -> tokens [EOS, c1..c125, BOS, PAD] = 128 total
        # The first 95 tokens (EOS + first 94 chars) may overlap if charset ordering matches
        old_embed_key = "text_embed.embedding.weight"
        if old_embed_key in pretrained_sd:
            old_embed = pretrained_sd[old_embed_key]  # (97, 384)
            new_embed = model.text_embed.embedding.weight.data  # (128, 384)
            # Copy EOS (index 0)
            new_embed[0] = old_embed[0]
            # Copy BOS and PAD from end positions
            new_embed[-2] = old_embed[-2]  # BOS
            new_embed[-1] = old_embed[-1]  # PAD
            print(f"  Transferred EOS/BOS/PAD embeddings from pretrained")

    return model


# ---------------------------------------------------------------------------
# Permutation training (PARSeq's core innovation)
# ---------------------------------------------------------------------------

class PermutationTrainer:
    """
    Implements PARSeq's permutation-aware training loss.

    During training, the model is trained with multiple autoregressive orderings
    (permutations) of the target sequence. This teaches the model bidirectional
    context — critical for non-autoregressive inference.
    """

    def __init__(self, perm_num=6, perm_forward=True, perm_mirrored=True):
        self.perm_num = perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.max_gen_perms = perm_num // 2 if perm_mirrored else perm_num
        self.rng = np.random.default_rng()

    def gen_tgt_perms(self, tgt, device):
        """Generate target permutations for training."""
        max_num_chars = tgt.shape[1] - 2  # exclude BOS and EOS
        if max_num_chars == 1:
            return torch.arange(3, device=device).unsqueeze(0)

        perms = [torch.arange(max_num_chars, device=device)] if self.perm_forward else []
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)

        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            if len(perms) == 0:
                perms = perm_pool[self.rng.choice(len(perm_pool), size=num_gen_perms, replace=False)]
            else:
                perms = torch.stack(perms)
                remaining = num_gen_perms - perms.shape[0]
                if remaining > 0 and len(perm_pool) > 0:
                    idx = self.rng.choice(len(perm_pool), size=min(remaining, len(perm_pool)), replace=False)
                    perms = torch.cat([perms, perm_pool[idx]])
        else:
            perms.extend([torch.randperm(max_num_chars, device=device) for _ in range(num_gen_perms - len(perms))])
            perms = torch.stack(perms)

        if self.perm_mirrored:
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)

        # Prepend BOS position (0) and append EOS position (max_num_chars + 1)
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)

        # Second permutation is always the reverse
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=device)

        return perms

    def generate_attn_masks(self, perm, device):
        """Convert a permutation into attention masks for the decoder."""
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=device)] = True
        query_mask = mask[1:, :-1]
        return content_mask, query_mask

    def compute_loss(self, model, images, labels, tokenizer, device):
        """
        Compute PARSeq permutation training loss.

        This is the exact loss from baudm/parseq system.py.
        """
        tgt = tokenizer.encode(labels, device)
        memory = model.encode(images)

        tgt_perms = self.gen_tgt_perms(tgt, device)
        tgt_in = tgt[:, :-1]    # input: [BOS, c1, c2, ..., cn]
        tgt_out = tgt[:, 1:]    # target: [c1, c2, ..., cn, EOS]
        tgt_padding_mask = (tgt_in == tokenizer.pad_id) | (tgt_in == tokenizer.eos_id)

        loss = 0
        loss_numel = 0
        n = (tgt_out != tokenizer.pad_id).sum().item()

        for i, perm in enumerate(tgt_perms):
            tgt_mask, query_mask = self.generate_attn_masks(perm, device)
            out = model.decode(
                tgt_in, memory, tgt_mask, tgt_padding_mask,
                tgt_query_mask=query_mask,
            )
            logits = model.head(out).flatten(end_dim=1)
            loss += n * F.cross_entropy(logits, tgt_out.flatten(), ignore_index=tokenizer.pad_id)
            loss_numel += n

            # After first two perms (forward + reverse), remove EOS from targets
            if i == 1:
                tgt_out = torch.where(tgt_out == tokenizer.eos_id, tokenizer.pad_id, tgt_out)
                n = (tgt_out != tokenizer.pad_id).sum().item()

        loss /= loss_numel
        return loss


# ---------------------------------------------------------------------------
# Evaluation
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
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
        prev_row = curr_row
    return prev_row[-1]


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device):
    """Evaluate model on validation set. Returns dict with CER, WER, accuracy, diacritical accuracy."""
    model.eval()
    total_chars = 0
    total_char_errors = 0
    total_words = 0
    total_word_errors = 0
    total_correct = 0
    total_samples = 0
    diac_total = 0
    diac_correct = 0

    for images, labels in val_loader:
        images = images.to(device)
        # Inference: forward pass with tokenizer for AR decoding
        logits = model.forward(tokenizer, images)
        probs = logits.softmax(-1)
        preds, _ = tokenizer.decode(probs)

        for pred_text, true_text in zip(preds, labels):
            total_chars += len(true_text)
            total_char_errors += levenshtein_distance(pred_text, true_text)
            total_words += max(len(true_text.split()), 1)
            total_word_errors += levenshtein_distance(pred_text.split(), true_text.split())
            total_samples += 1
            if pred_text == true_text:
                total_correct += 1

            # Diacritical accuracy (aligned character comparison)
            for j, c in enumerate(true_text):
                if c in DIACRITICAL_CHARS:
                    diac_total += 1
                    if j < len(pred_text) and pred_text[j] == c:
                        diac_correct += 1

    model.train()
    return {
        "cer": total_char_errors / max(total_chars, 1),
        "wer": total_word_errors / max(total_words, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "diacritical_accuracy": diac_correct / max(diac_total, 1),
        "num_samples": total_samples,
    }


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, step, metrics, path):
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "metrics": metrics,
    }
    torch.save(state, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune PARSeq for Nordic OCR")

    # Data
    parser.add_argument("--auto-labeled", type=str, required=True)
    parser.add_argument("--spot-check", type=str, default=None)
    parser.add_argument("--val-dir", type=str, default=None)
    parser.add_argument("--spot-check-oversample", type=int, default=4)

    # Model
    parser.add_argument("--pretrained", type=str, default="parseq",
                        choices=list(PRETRAINED_URLS.keys()))
    parser.add_argument("--max-label-length", type=int, default=128,
                        help="Max text length in characters")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=768,
                        help="Per-GPU batch size (default: 768 for single RTX 6000 PRO at 32x384 images)")
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-pct", type=float, default=0.075)
    parser.add_argument("--perm-num", type=int, default=6)
    parser.add_argument("--grad-clip", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=8)

    # Output
    parser.add_argument("--output-dir", type=str, default="output/parseq_nordic")
    parser.add_argument("--val-interval", type=int, default=1000)

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    img_size = PARSEQ_CONFIG["img_size"]

    # Distributed setup
    distributed = args.local_rank >= 0 or "RANK" in os.environ
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True

    # Blackwell optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    writer = None
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.output_dir, "logs"))
        except ImportError:
            print("tensorboard not installed — skipping TB logging")

    # Tokenizer
    tokenizer = NordicTokenizer()
    if is_main:
        print(f"Charset: {len(tokenizer.charset)} chars, vocab: {tokenizer.vocab_size}, "
              f"output classes: {tokenizer.num_output_classes}")

    # Dataset
    train_dataset = build_train_dataset(
        auto_labeled_dir=args.auto_labeled,
        spot_check_dir=args.spot_check,
        tokenizer=tokenizer,
        img_size=img_size,
        max_label_len=args.max_label_length,
        spot_check_oversample=args.spot_check_oversample,
    )

    if args.val_dir:
        val_dataset = build_val_dataset(
            args.val_dir, tokenizer=tokenizer, img_size=img_size,
            max_label_len=args.max_label_length,
        )
    else:
        val_size = max(1000, int(0.05 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    # DataLoaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn,
    )

    if is_main:
        print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"Val: {len(val_dataset)} samples")

    # Model
    model = load_parseq_model(tokenizer, args.pretrained, args.max_label_length)
    model = model.to(device)
    model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Permutation trainer
    perm_trainer = PermutationTrainer(perm_num=args.perm_num)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_pct)
    scaler = GradScaler("cuda")

    # Training loop
    best_cer = float("inf")
    global_step = 0

    world_size = torch.distributed.get_world_size() if distributed else 1
    if is_main:
        print(f"\nTraining on {device}")
        print(f"  Epochs: {args.epochs}, Batch/GPU: {args.batch_size}, "
              f"Effective batch: {args.batch_size * world_size}, LR: {args.lr}")
        print(f"  Permutations: {args.perm_num}, Total steps: {total_steps}")
        print(f"  AMP: bf16, TF32: enabled, torch.compile: enabled\n")

    raw_model = model.module if distributed else model

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        for images, labels in pbar:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = perm_trainer.compute_loss(raw_model, images, labels, tokenizer, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if is_main:
                if writer:
                    writer.add_scalar("train/loss", loss.item(), global_step)

            # Periodic validation
            if global_step % args.val_interval == 0 and is_main:
                metrics = evaluate(raw_model, val_loader, tokenizer, device)
                print(f"\n  Step {global_step}: CER={metrics['cer']:.4f} "
                      f"Acc={metrics['accuracy']:.4f} Diac={metrics['diacritical_accuracy']:.4f}")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        if writer:
                            writer.add_scalar(f"val/{k}", v, global_step)
                if metrics["cer"] < best_cer:
                    best_cer = metrics["cer"]
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics,
                                    os.path.join(args.output_dir, "best.pth"))
                    print(f"    New best CER: {best_cer:.4f} — saved!")

        # End of epoch
        if is_main:
            avg_loss = epoch_loss / max(len(train_loader), 1)
            metrics = evaluate(raw_model, val_loader, tokenizer, device)
            print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f} CER={metrics['cer']:.4f} "
                  f"Acc={metrics['accuracy']:.4f} Diac={metrics['diacritical_accuracy']:.4f}")

            if metrics["cer"] < best_cer:
                best_cer = metrics["cer"]
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics,
                                os.path.join(args.output_dir, "best.pth"))

            save_checkpoint(model, optimizer, scheduler, epoch, global_step, metrics,
                            os.path.join(args.output_dir, f"epoch_{epoch+1}.pth"))

    if is_main:
        save_checkpoint(model, optimizer, scheduler, args.epochs - 1, global_step, metrics,
                        os.path.join(args.output_dir, "final.pth"))
        if writer:
            writer.close()
        print(f"\nDone. Best CER: {best_cer:.4f}. Checkpoints in: {args.output_dir}")

    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
