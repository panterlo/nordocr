"""
SVTRv2 fine-tuning for Nordic text recognition via OpenOCR.

SVTRv2 (Scene Visual Text Recognition v2) is a CTC-based recognition model
that uses attention-enhanced CTC decoding (RCTC). Accepted at ICCV 2025.

Trade-off vs PARSeq:
  + CTC decoding is inherently parallel (no autoregressive loop)
  + CTC produces simpler TensorRT computational graphs (no onnxsim needed)
  + Multi-Size Resizing handles variable text line widths natively
  - Newer model, less battle-tested in production deployments
  - CTC has inherent limitation on repeated characters (mitigated by blank tokens)

Prerequisites:
    pip install lmdb
    git clone https://github.com/Topdu/OpenOCR.git
    pip install -r OpenOCR/requirements.txt

Data preparation:
    # 1. Split dataset
    python tools/split_dataset.py \
        --input D:/TrainingData/auto_labeled \
        --output D:/TrainingData/splits --copy

    # 2. Convert to LMDB
    python tools/tsv_to_lmdb.py \
        --input D:/TrainingData/splits/train \
        --output D:/TrainingData/lmdb/train \
        --charset-file recognize/nordic_dict.txt

    python tools/tsv_to_lmdb.py \
        --input D:/TrainingData/splits/val \
        --output D:/TrainingData/lmdb/val \
        --charset-file recognize/nordic_dict.txt

Usage:
    # Generate config and train
    python train_svtrv2.py \
        --train-lmdb D:/TrainingData/lmdb/train \
        --val-lmdb D:/TrainingData/lmdb/val \
        --output-dir output/svtrv2_nordic

    # Multi-GPU
    torchrun --nproc_per_node=2 train_svtrv2.py \
        --train-lmdb D:/TrainingData/lmdb/train \
        --val-lmdb D:/TrainingData/lmdb/val \
        --output-dir output/svtrv2_nordic

    # With pretrained weights
    python train_svtrv2.py \
        --train-lmdb D:/TrainingData/lmdb/train \
        --val-lmdb D:/TrainingData/lmdb/val \
        --pretrained openocr_svtrv2_rctc.pth \
        --output-dir output/svtrv2_nordic
"""

import argparse
import os
import sys
import yaml
from pathlib import Path


PRETRAINED_URLS = {
    "svtrv2_rctc": "https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_svtrv2_ch.pth",
}

# Path to Nordic character dictionary (one char per line, no blank token — CTC adds it)
NORDIC_DICT_PATH = str(Path(__file__).parent / "nordic_dict.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SVTRv2 for Nordic OCR")
    parser.add_argument("--train-lmdb", type=str, required=True,
                        help="Path to training LMDB directory")
    parser.add_argument("--val-lmdb", type=str, required=True,
                        help="Path to validation LMDB directory")
    parser.add_argument("--spot-check-lmdb", type=str, default=None,
                        help="Path to spot-check LMDB (gold standard, oversampled 4x)")
    parser.add_argument("--spot-check-oversample", type=int, default=4,
                        help="Spot-check oversampling factor (default: 4)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained SVTRv2 weights (.pth)")
    parser.add_argument("--output-dir", type=str, default="output/svtrv2_nordic")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Base batch size at 128x32 resolution (scales down for wider images)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (3e-4 default, lower for stability with variable-width)")
    parser.add_argument("--grad-clip", type=float, default=5.0,
                        help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--max-label-len", type=int, default=100,
                        help="Max label length (100 for full-width lines at 1792px)")
    parser.add_argument("--max-ratio", type=int, default=56,
                        help="Max width/height ratio (56 = 1792px at height 32, full variable-width)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--variant", type=str, default="base",
                        choices=["tiny", "small", "base"],
                        help="SVTRv2 size variant")
    parser.add_argument("--decoder", type=str, default="rctc",
                        choices=["ctc", "rctc"],
                        help="Decoder type (rctc recommended, ctc for simpler ONNX export)")
    parser.add_argument("--openocr-dir", type=str, default=None,
                        help="Path to cloned OpenOCR repo (auto-detected if on PYTHONPATH)")
    parser.add_argument("--config-only", action="store_true",
                        help="Only generate YAML config, don't launch training")
    return parser.parse_args()


# Model variant configurations
VARIANTS = {
    "tiny": {
        "dims": [64, 128, 256],
        "depths": [3, 6, 3],
        "num_heads": [2, 4, 8],
    },
    "small": {
        "dims": [96, 192, 320],
        "depths": [3, 6, 6],
        "num_heads": [3, 6, 10],
    },
    "base": {
        "dims": [128, 256, 384],
        "depths": [6, 6, 6],
        "num_heads": [4, 8, 12],
    },
}


def _build_data_dir_list(args):
    """Build training data directory list with spot-check oversampling."""
    dirs = [args.train_lmdb]
    if args.spot_check_lmdb:
        for _ in range(args.spot_check_oversample):
            dirs.append(args.spot_check_lmdb)
    return dirs


def generate_config(args):
    """Generate OpenOCR-compatible YAML config for SVTRv2 Nordic fine-tuning."""
    variant = VARIANTS[args.variant]
    use_rctc = args.decoder == "rctc"

    config = {
        "Global": {
            "device": "gpu",
            "epoch_num": args.epochs,
            "log_smooth_window": 20,
            "print_batch_step": 100,
            "output_dir": args.output_dir,
            "save_epoch_step": [max(1, args.epochs - 5), 1],
            "eval_epoch_step": [0, 1],
            "eval_batch_step": [0, 500],
            "cal_metric_during_train": True,
            "pretrained_model": args.pretrained,
            "checkpoints": None,
            "use_tensorboard": False,
            "character_dict_path": NORDIC_DICT_PATH,
            "max_text_length": args.max_label_len,
            "use_space_char": False,
            "save_res_path": os.path.join(args.output_dir, "predicts.txt"),
            "use_amp": True,
            "amp_dtype": "float16",
            "use_tf32": True,
            "grad_clip_val": args.grad_clip,
        },
        "Optimizer": {
            "name": "AdamW",
            "lr": args.lr,
            "weight_decay": 0.05,
            "filter_bias_and_bn": True,
        },
        "LRScheduler": {
            "name": "OneCycleLR",
            "warmup_epoch": 1.5,
            "cycle_momentum": False,
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": "SVTRv2",
            "Transform": None,
            "Encoder": {
                "name": "SVTRv2LNConvTwo33",
                "use_pos_embed": False,
                "dims": variant["dims"],
                "depths": variant["depths"],
                "num_heads": variant["num_heads"],
                "mixer": [
                    ["Conv", "Conv", "Conv", "Conv", "Conv", "Conv"],
                    ["Conv", "Conv", "FGlobal", "Global", "Global", "Global"],
                    ["Global", "Global", "Global", "Global", "Global", "Global"],
                ],
                "local_k": [[5, 5], [5, 5], [-1, -1]],
                "sub_k": [[1, 1], [2, 1], [-1, -1]],
                "last_stage": not use_rctc,
                "feat2d": use_rctc,
            },
            "Decoder": {
                "name": "RCTCDecoder" if use_rctc else "CTCDecoder",
                # out_channels set automatically from character dictionary
            },
        },
        "Loss": {
            "name": "CTCLoss",
            "zero_infinity": True,
        },
        "PostProcess": {
            "name": "CTCLabelDecode",
            "character_dict_path": NORDIC_DICT_PATH,
            "use_space_char": False,
        },
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
            "is_filter": True,
        },
        "Train": {
            "dataset": {
                "name": "RatioDataSetTVResize",
                "ds_width": True,
                "padding": False,
                "data_dir_list": _build_data_dir_list(args),
                "transforms": [
                    {"DecodeImagePIL": {"img_mode": "RGB"}},
                    {"PARSeqAugPIL": None},
                    {"CTCLabelEncode": {
                        "character_dict_path": NORDIC_DICT_PATH,
                        "use_space_char": False,
                        "max_text_length": args.max_label_len,
                    }},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                ],
            },
            "sampler": {
                "name": "RatioSampler",
                "scales": [[128, 32]],
                "first_bs": args.batch_size,
                "fix_bs": False,
                "divided_factor": [4, 16],
                "is_training": True,
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": args.batch_size,
                "drop_last": True,
                "max_ratio": args.max_ratio,
                "num_workers": args.num_workers,
            },
        },
        "Eval": {
            "dataset": {
                "name": "RatioDataSetTVResize",
                "ds_width": True,
                "padding": False,
                "data_dir_list": [args.val_lmdb],
                "transforms": [
                    {"DecodeImagePIL": {"img_mode": "RGB"}},
                    {"CTCLabelEncode": {
                        "character_dict_path": NORDIC_DICT_PATH,
                        "use_space_char": False,
                        "max_text_length": args.max_label_len,
                    }},
                    {"KeepKeys": {"keep_keys": ["image", "label", "length"]}},
                ],
            },
            "sampler": {
                "name": "RatioSampler",
                "scales": [[128, 32]],
                "first_bs": args.batch_size,
                "fix_bs": False,
                "divided_factor": [4, 16],
                "is_training": False,
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": args.batch_size,
                "max_ratio": args.max_ratio,
                "num_workers": args.num_workers,
            },
        },
    }

    # Trim mixer lists to match actual depths (variants with depth < 6)
    for i, depth in enumerate(variant["depths"]):
        mixer_row = config["Architecture"]["Encoder"]["mixer"][i]
        config["Architecture"]["Encoder"]["mixer"][i] = mixer_row[:depth]

    return config


def find_openocr(openocr_dir=None):
    """Find and validate OpenOCR installation."""
    # Check explicit path
    if openocr_dir:
        p = Path(openocr_dir)
        if (p / "tools" / "train_rec.py").exists():
            return p
        print(f"WARNING: OpenOCR not found at {openocr_dir}")

    # Check if already importable
    try:
        import openrec
        return Path(openrec.__file__).parent.parent
    except ImportError:
        pass

    # Check common locations
    for candidate in [
        Path.home() / "OpenOCR",
        Path(__file__).parent.parent.parent / "OpenOCR",
        Path("OpenOCR"),
    ]:
        if (candidate / "tools" / "train_rec.py").exists():
            return candidate

    return None


def download_pretrained(name, output_dir):
    """Download pretrained weights if a known name is given."""
    if name in PRETRAINED_URLS:
        url = PRETRAINED_URLS[name]
        output_path = os.path.join(output_dir, f"{name}.pth")
        if os.path.exists(output_path):
            print(f"Pretrained weights already exist: {output_path}")
            return output_path
        print(f"Downloading pretrained weights: {name}")
        import urllib.request
        os.makedirs(output_dir, exist_ok=True)
        urllib.request.urlretrieve(url, output_path)
        print(f"  Saved to: {output_path}")
        return output_path
    return name  # Assume it's already a file path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Download pretrained if needed
    if args.pretrained and args.pretrained in PRETRAINED_URLS:
        args.pretrained = download_pretrained(args.pretrained, args.output_dir)

    # Generate config
    config = generate_config(args)
    config_path = os.path.join(args.output_dir, "config.yml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)

    print(f"Config written: {config_path}")
    print(f"  Variant: SVTRv2-{args.variant} + {args.decoder.upper()} decoder")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Train LMDB: {args.train_lmdb}")
    print(f"  Val LMDB: {args.val_lmdb}")
    if args.pretrained:
        print(f"  Pretrained: {args.pretrained}")

    if args.config_only:
        print(f"\nConfig-only mode. To train, run:")
        print(f"  python OpenOCR/tools/train_rec.py --c {config_path}")
        return

    # Find OpenOCR
    openocr_path = find_openocr(args.openocr_dir)
    if openocr_path is None:
        print(f"\nERROR: OpenOCR not found. Install it first:")
        print(f"  git clone https://github.com/Topdu/OpenOCR.git")
        print(f"  pip install -r OpenOCR/requirements.txt")
        print(f"\nOr run with --config-only and launch manually:")
        print(f"  python train_svtrv2.py --config-only ...")
        print(f"  python OpenOCR/tools/train_rec.py --c {config_path}")
        sys.exit(1)

    print(f"  OpenOCR: {openocr_path}")

    # Add OpenOCR to path and launch training
    sys.path.insert(0, str(openocr_path))
    os.chdir(str(openocr_path))

    # Import and run OpenOCR trainer
    from tools.train_rec import main as openocr_main
    sys.argv = ["train_rec.py", "--c", config_path]
    openocr_main()


if __name__ == "__main__":
    main()
