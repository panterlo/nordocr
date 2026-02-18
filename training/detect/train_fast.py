"""
FAST fine-tuning for Nordic document text detection via docTR.

FAST (Faster Arbitrarily-Shaped Text Detector) uses a minimalist 1-channel
kernel representation with GPU-parallel post-processing. ~3x faster than
DBNet++ at comparable accuracy. Default detector in docTR since 2024.

Prerequisites:
    pip install python-doctr[torch]
    # Or for GPU: pip install python-doctr[torch] --extra-index-url https://download.pytorch.org/whl/cu121

Data format (docTR detection JSON):
    {
      "img_001.png": {
        "img_dimensions": [height, width],
        "img_hash": "sha256_hash",
        "polygons": [
          [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
          ...
        ]
      }
    }
    Polygon coordinates are normalized to [0, 1] relative to image dimensions.

Usage:
    # Single GPU
    python train_fast.py \
        --train-path /data/DetectionData/train \
        --val-path /data/DetectionData/val \
        --output-dir output/detect_fast

    # Multi-GPU (2x RTX 6000 PRO Blackwell)
    torchrun --nproc_per_node=2 train_fast.py \
        --train-path /data/DetectionData/train \
        --val-path /data/DetectionData/val \
        --output-dir output/detect_fast

    # FAST variants: fast_tiny (8.5M), fast_small (9.7M), fast_base (10.6M)
    python train_fast.py --variant fast_base ...

Data directory structure:
    train/
        images/
            img_001.png
            img_002.png
        labels.json
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm


# Pretrained weight URLs from official FAST repo (IC17-MLT trained)
PRETRAINED_URLS = {
    "fast_tiny": "https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth",
    "fast_small": "https://github.com/czczup/FAST/releases/download/release/fast_small_ic17mlt_640.pth",
    "fast_base": "https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train FAST for Nordic text detection")
    parser.add_argument("--train-path", type=str, required=True,
                        help="Training directory (images/ + labels.json)")
    parser.add_argument("--val-path", type=str, required=True,
                        help="Validation directory (images/ + labels.json)")
    parser.add_argument("--output-dir", type=str, default="output/detect_fast")
    parser.add_argument("--variant", type=str, default="fast_base",
                        choices=["fast_tiny", "fast_small", "fast_base"],
                        help="FAST model variant")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24,
                        help="Per-GPU batch size (default: 24 for RTX 6000 PRO with ~30GB free)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Training image size (square)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: True)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (disables --pretrained)")
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--val-interval", type=int, default=5,
                        help="Validate every N epochs")

    # Distributed
    parser.add_argument("--local_rank", type=int, default=-1)

    return parser.parse_args()


def train_with_doctr(args):
    """Train FAST using docTR's built-in training infrastructure."""
    try:
        from doctr.models import detection
        from doctr.datasets import DetectionDataset
        from doctr.transforms import Resize
    except ImportError:
        print("ERROR: docTR not installed. Install it first:")
        print("  pip install 'python-doctr[torch]'")
        sys.exit(1)

    # Distributed setup
    distributed = args.local_rank >= 0 or "RANK" in os.environ
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        is_main = local_rank == 0
        world_size = torch.distributed.get_world_size()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
        world_size = 1

    # Blackwell optimizations
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if is_main:
        print(f"Training on {device} (world_size={world_size})")

    # Load model
    if args.resume:
        model = detection.__dict__[args.variant](pretrained=False, exportable=False)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
        if is_main:
            print(f"Resumed from: {args.resume}")
    else:
        model = detection.__dict__[args.variant](
            pretrained=args.pretrained, exportable=False
        )
        if is_main:
            print(f"Loaded {args.variant} (pretrained={args.pretrained})")

    model = model.to(device)
    model = torch.compile(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Datasets
    train_dataset = DetectionDataset(
        img_folder=os.path.join(args.train_path, "images"),
        label_path=os.path.join(args.train_path, "labels.json"),
        sample_transforms=Resize((args.img_size, args.img_size)),
    )
    val_dataset = DetectionDataset(
        img_folder=os.path.join(args.val_path, "images"),
        label_path=os.path.join(args.val_path, "labels.json"),
        sample_transforms=Resize((args.img_size, args.img_size)),
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if is_main:
        print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
        print(f"Val: {len(val_dataset)} images")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=len(train_loader) * args.epochs,
        pct_start=0.1,
    )
    scaler = GradScaler("cuda")

    # Training loop
    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)

    if is_main:
        print(f"\nTraining config:")
        print(f"  Epochs: {args.epochs}, Batch/GPU: {args.batch_size}, "
              f"Effective batch: {args.batch_size * world_size}")
        print(f"  LR: {args.lr}, Image size: {args.img_size}x{args.img_size}")
        print(f"  AMP: bf16, TF32: enabled, torch.compile: enabled\n")

    raw_model = model.module if distributed else model

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        for images, targets in pbar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(images, targets)["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)

        # Validation
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        val_loss += model(images, targets)["loss"].item()
            val_loss /= max(len(val_loader), 1)

            if is_main:
                print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f}")

                # Save checkpoints
                state = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                }

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(state, os.path.join(args.output_dir, "best.pth"))
                    print(f"  New best val_loss: {best_loss:.4f}")

                if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                    torch.save(state, os.path.join(args.output_dir, f"epoch_{epoch+1}.pth"))
        elif is_main:
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")

    # Export to ONNX (main process only)
    if is_main:
        print(f"\nExporting best model to ONNX...")
        raw_model.load_state_dict(
            torch.load(os.path.join(args.output_dir, "best.pth"),
                        map_location=device, weights_only=False)["model"]
        )
        export_model = detection.__dict__[args.variant](pretrained=False, exportable=True)
        export_model.load_state_dict(raw_model.state_dict())
        export_model.eval()

        try:
            from doctr.models.utils import export_model_to_onnx
            dummy = torch.rand(1, 3, args.img_size, args.img_size)
            onnx_path = os.path.join(args.output_dir, f"{args.variant}_nordic.onnx")
            export_model_to_onnx(export_model, model_name=onnx_path, dummy_input=dummy)
            print(f"  ONNX saved: {onnx_path}")
        except Exception as e:
            print(f"  ONNX export failed: {e}")
            print(f"  You can export manually with exportable=True")

        print(f"\nDone. Best val_loss: {best_loss:.4f}")
        print(f"Checkpoints in: {args.output_dir}")

    if distributed:
        torch.distributed.destroy_process_group()


def main():
    args = parse_args()

    is_main = int(os.environ.get("LOCAL_RANK", args.local_rank)) <= 0
    if is_main:
        print(f"FAST Detection Training")
        print(f"  Variant: {args.variant}")
        print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
        print(f"  Image size: {args.img_size}x{args.img_size}")
        print(f"  Train: {args.train_path}")
        print(f"  Val: {args.val_path}")

    train_with_doctr(args)


if __name__ == "__main__":
    main()
