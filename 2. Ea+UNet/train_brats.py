"""
train_brats.py  —  Entry point for EA-ResNetUNet v2 training (Laptop Mode).

=============================================================================
LAPTOP-OPTIMIZED DEFAULTS  (RTX 4050, 6 GB VRAM, ~2.5 GB free RAM)
=============================================================================

    --img_size     192    (vs 256 in v1)  — saves 1.8× VRAM per batch
    --batch_size   auto   → 8 at 192px   — safe on 6 GB VRAM with AMP
    --num_workers  auto   → 2            — prevents OOM from NIfTI caching
    --max_cases    500                   — use 500 of ~1251 BraTS cases
    --atiss_k      10                    — select top-10 slices per case
    --max_epochs   50                    — enough to see convergence signal
    --base_lr      0.01                  — standard SGD lr for medical seg

Total training slices: 500 × 0.9 × 10 = 4500 (train) + 500 (val)
Time per epoch (RTX 4050, bs=8, AMP): ~3-5 min
Total time for 50 epochs: ~2.5 - 4 hours

=============================================================================
HOW TO RUN
=============================================================================

# Basic (auto-detect everything):
    python train_brats.py

# Custom path:
    python train_brats.py --root_path "D:/MaS-TransUNet/BraTS2021_Training_Data"

# More epochs after sanity check:
    python train_brats.py --max_epochs 100 --resume true

# If CUDA OOM occurs:
    python train_brats.py --img_size 160 --batch_size 8

=============================================================================
"""

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models import ResNetUNet
from utils.performance import (
    auto_inference_batch_size,
    auto_num_workers,
    auto_train_batch_size,
    get_gpu_info,
)
from trainer_brats import trainer_brats
from utils.visualize import plot_training_curve


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# ---- Dataset path resolution ----
# Priority: preprocessed 2D (if it exists) > raw 3D BraTS
DEFAULT_RAW_ROOT_PATH       = os.path.join(REPO_ROOT, "BraTS2021_Training_Data")
DEFAULT_PROCESSED_ROOT_PATH = os.path.join(BASE_DIR, "processed_brats_2d")


def resolve_default_root_path():
    """Prefer preprocessed 2D dataset (faster I/O) if already prepared."""
    if os.path.isdir(DEFAULT_PROCESSED_ROOT_PATH):
        return DEFAULT_PROCESSED_ROOT_PATH
    return DEFAULT_RAW_ROOT_PATH


DEFAULT_ROOT_PATH = resolve_default_root_path()


def str2bool(value):
    """Parse boolean CLI arguments (true/false, 1/0, yes/no, y/n)."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


# =============================================================================
# CLI Argument Parser
# =============================================================================

parser = argparse.ArgumentParser(
    description=(
        "Train Edge-Attention ResNet-UNet v2 on BraTS.\n\n"
        "Laptop Mode: 500 cases, ATISS k=10, img_size=192, batch_size=8.\n"
        "Loss: 0.5*CE + 0.5*Dice + 0.3*EdgeBCEDice"
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# ---- Dataset ----
data_group = parser.add_argument_group("Dataset")
data_group.add_argument(
    "--root_path", type=str, default=DEFAULT_ROOT_PATH,
    help="BraTS dataset root (preprocessed 2D or raw 3D NIfTI)"
)
data_group.add_argument(
    "--dataset", type=str, default="brats",
    help="Dataset name (only 'brats' supported)"
)
data_group.add_argument(
    "--num_classes", type=int, default=4,
    help="Segmentation classes: 0=BG, 1=NCR/NET, 2=ED, 3=ET"
)
data_group.add_argument(
    "--input_channels", type=int, default=4,
    help="MRI modalities: T1, T1ce, T2, FLAIR = 4"
)
data_group.add_argument(
    "--val_split", type=float, default=0.1,
    help="Fraction of cases for validation (0.1 = 10%%)"
)
data_group.add_argument(
    "--max_cases", type=int, default=500,
    help=(
        "Max number of BraTS cases to use. "
        "500 → ~4500 train slices (laptop mode). "
        "Set to 0 or -1 for all cases."
    )
)
data_group.add_argument(
    "--atiss_k", type=int, default=10,
    help=(
        "ATISS: number of informative slices selected per 3D case. "
        "k=10 gives 10 slices/case × 450 cases = 4500 train slices. "
        "Set to 0 to disable ATISS and use all slices."
    )
)

# ---- Training ----
train_group = parser.add_argument_group("Training")
train_group.add_argument(
    "--max_epochs", type=int, default=50,
    help="Total training epochs (50 = enough to verify learning)"
)
train_group.add_argument(
    "--batch_size", type=int, default=0,
    help="Batch size (0 = auto: 8 at 192px on RTX 4050)"
)
train_group.add_argument(
    "--base_lr", type=float, default=0.01,
    help="Initial SGD learning rate"
)
train_group.add_argument(
    "--img_size", type=int, default=192,
    help="Input resolution. 192 recommended for 6 GB VRAM."
)
train_group.add_argument(
    "--seed", type=int, default=1234,
    help="Global random seed"
)
train_group.add_argument(
    "--deterministic", type=int, default=0,
    help="CuDNN deterministic mode (0=faster benchmark mode)"
)

# ---- Hardware ----
hw_group = parser.add_argument_group("Hardware")
hw_group.add_argument(
    "--n_gpu", type=int, default=1,
    help="Number of GPUs (1 for laptop)"
)
hw_group.add_argument(
    "--use_amp", type=str2bool, default=True,
    help=(
        "Mixed precision (AMP, float16). STRONGLY recommended on laptop "
        "— reduces VRAM usage by ~40%% and speeds up by ~30%%."
    )
)
hw_group.add_argument(
    "--num_workers", type=int, default=-1,
    help="DataLoader workers (-1 = auto: 2 on laptop)"
)
hw_group.add_argument(
    "--channels_last", type=str2bool, default=True,
    help="channels_last memory format — faster convolutions on CUDA"
)
hw_group.add_argument(
    "--use_tf32", type=str2bool, default=True,
    help="TF32 on RTX 4050 (~2× faster matmul, tiny precision loss)"
)

# ---- Output ----
out_group = parser.add_argument_group("Output")
out_group.add_argument(
    "--name", type=str, default="EA_ResNetUNet_v2_brats",
    help="Experiment name (output directory)"
)
out_group.add_argument(
    "--resume", type=str2bool, default=True,
    help="Resume from last checkpoint if available"
)
out_group.add_argument(
    "--save_every_steps", type=int, default=200,
    help=(
        "Mid-epoch checkpoint interval (steps). "
        "200 is safe for ~562 steps/epoch at bs=8."
    )
)

args = parser.parse_args()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    start_time            = time.time()
    current_date_and_time = datetime.now()

    print("=" * 65)
    print("  Edge-Attention ResNet-UNet v2  —  LAPTOP TRAINING MODE")
    print("=" * 65)
    print(f"  Torch version   = {torch.__version__}")
    print(f"  CUDA version    = {torch.version.cuda}")
    print(f"  Started at      = {current_date_and_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ---- Deterministic vs. benchmark mode ----
    if not args.deterministic:
        # benchmark=True: cuDNN finds fastest algorithm — ~10-15% speedup.
        # Requires fixed input size (which we have).
        cudnn.benchmark   = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark   = False
        cudnn.deterministic = True

    # ---- Seed all RNGs ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Device ----
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Auto-detect settings ----
    if args.num_workers < 0:
        # Cap at 2 for laptop (protect RAM)
        args.num_workers = auto_num_workers(max_cap=2)

    if args.batch_size <= 0:
        # RTX 4050 safe default: bs=8 at 192px
        args.batch_size = auto_train_batch_size(args.img_size)

    # ---- Normalize max_cases ----
    if args.max_cases <= 0:
        args.max_cases = None   # None = use all cases

    # ---- GPU optimizations ----
    if args.device.type == "cuda":
        if args.use_tf32:
            # TF32: Ada Lovelace architecture supports this natively.
            # ~2× faster matrix multiply with tiny (0.1%) precision loss.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32        = True
            torch.set_float32_matmul_precision("high")

    # ---- Print hardware info ----
    get_gpu_info()
    print()

    # ---- Dataset config ----
    dataset_config = {
        "brats": {
            "root_path":      DEFAULT_ROOT_PATH,
            "num_classes":    4,
            "input_channels": 4,
        }
    }

    dataset_name        = args.dataset
    args.root_path      = args.root_path or dataset_config[dataset_name]["root_path"]
    args.num_classes    = dataset_config[dataset_name]["num_classes"]
    args.input_channels = dataset_config[dataset_name]["input_channels"]

    # ---- Print training configuration ----
    print("Training Configuration:")
    print(f"  Dataset root    = {args.root_path}")
    print(f"  Max cases       = {args.max_cases} (None = all)")
    print(f"  ATISS k         = {args.atiss_k} slices/case")
    print(f"  Est. train slices = ~{int((args.max_cases or 1251) * (1-args.val_split) * args.atiss_k)}")
    print(f"  Image size      = {args.img_size}×{args.img_size}")
    print(f"  Batch size      = {args.batch_size}")
    print(f"  Num workers     = {args.num_workers}")
    print(f"  Max epochs      = {args.max_epochs}")
    print(f"  Base LR         = {args.base_lr}")
    print(f"  Val split       = {args.val_split}")
    print(f"  AMP (float16)   = {args.use_amp}")
    print(f"  TF32            = {args.use_tf32}")
    print(f"  channels_last   = {args.channels_last}")
    print(f"  Device          = {args.device}")

    if args.device.type != "cuda":
        print("\n  ⚠️  WARNING: CUDA not available. Training on CPU will be very slow.")
        print("     Ensure CUDA drivers are installed and pytorch-cuda is available.")

    # ---- Output directory ----
    args.exp      = args.name or f"EA_ResNetUNet_v2_{dataset_name}"
    snapshot_name = (
        f"EA_v2_epo{args.max_epochs}_bs{args.batch_size}"
        f"_cases{args.max_cases or 'all'}_k{args.atiss_k}_{args.img_size}"
    )
    snapshot_path = os.path.join(BASE_DIR, "outputs", args.exp, snapshot_name)
    os.makedirs(snapshot_path, exist_ok=True)

    print(f"  Output dir      = {snapshot_path}")
    print("=" * 65)

    # ---- Model ----
    model = ResNetUNet(
        num_classes=args.num_classes,
        input_channels=args.input_channels,
    ).to(args.device)

    if args.channels_last and args.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # ---- Print parameter count ----
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    edge_params      = sum(
        p.numel() for p in model.encoder.attention.parameters()
        if p.requires_grad
    )
    print(f"\nModel Parameters:")
    print(f"  Total           = {total_params:>12,}")
    print(f"  Trainable       = {trainable_params:>12,}")
    print(f"  Edge branch     = {edge_params:>12,}  ({100*edge_params/trainable_params:.1f}% of total)")
    print("=" * 65)
    print()

    # ---- Run training ----
    training_summary = trainer_brats(args, model, snapshot_path)

    # ---- Plot training curves ----
    try:
        curve_path = plot_training_curve(
            training_summary["log_csv"],
            os.path.join(snapshot_path, "training_curve.png"),
        )
        print(f"\nTraining curve saved → {curve_path}")
    except Exception as plot_err:
        print(f"\nWarning: Could not plot curve: {plot_err}")

    # ---- Final summary ----
    elapsed = time.time() - start_time
    print()
    print("=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Best val Dice   = {training_summary['best_val_dice']:.4f}")
    print(f"  Best val Loss   = {training_summary['best_val_loss']:.4f}")
    print(f"  Total time      = {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"  Checkpoints     → {snapshot_path}/")
    print()
    print("  Next steps:")
    print("    python test_brats.py --save_edge_maps true")
    print("    python evaluation_metrics/evaluate_metrics.py")
    print("=" * 65)
