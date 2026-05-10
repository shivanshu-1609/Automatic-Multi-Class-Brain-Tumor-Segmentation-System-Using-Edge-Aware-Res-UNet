"""
test_brats.py  —  Inference script for Edge-Attention ResNet-UNet v2.

This script generates segmentation predictions ONLY.
For metrics (Dice, HD95, IoU, Sensitivity, Specificity), run:
    python evaluation_metrics/evaluate_metrics.py

Changes vs ea+unet v1:
    - Checkpoint auto-detection updated for v2 output directory naming.
    - Model forward pass unchanged: returns (logits, lateral_edge).
      lateral_edge is discarded at inference — only logits are used.
    - Added --save_edge_maps flag to optionally save predicted edge maps
      as PNGs for visual inspection of what the model learned.
"""

import argparse
import glob
import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import BRATS_CLASS_NAMES, BraTSSliceDataset
from models import ResNetUNet
from utils.performance import auto_inference_batch_size, auto_num_workers
from utils.visualize import plot_predictions, save_class_masks, save_segmentation_mask


BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Phase2_Training is a sub-folder — processed data lives in the parent project
DEFAULT_PROCESSED_ROOT_PATH = os.path.join(REPO_ROOT, "ea+unet_v2", "processed_brats_2d")
if not os.path.isdir(DEFAULT_PROCESSED_ROOT_PATH):
    # Fallback: one level up (original project root)
    DEFAULT_PROCESSED_ROOT_PATH = os.path.join(os.path.dirname(REPO_ROOT), "ea+unet_v2", "processed_brats_2d")
if not os.path.isdir(DEFAULT_PROCESSED_ROOT_PATH):
    # Final fallback: sibling of Phase2_Training
    DEFAULT_PROCESSED_ROOT_PATH = os.path.join(REPO_ROOT, "processed_brats_2d")

DEFAULT_ROOT_PATH  = DEFAULT_PROCESSED_ROOT_PATH
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "inference_outputs")


def str2bool(value):
    """Parse boolean CLI arguments."""
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
        "Inference with Edge-Attention ResNet-UNet v2 on BraTS.\n"
        "Generates predictions (masks, overlays, eval pairs).\n"
        "Does NOT compute metrics — run evaluate_metrics.py afterwards."
    )
)

# Dataset and model
parser.add_argument("--root_path",      type=str,      default=DEFAULT_ROOT_PATH)
parser.add_argument("--checkpoint",     type=str,      default="",
                    help="Path to trained model weights (.pth). Auto-detected if empty.")
parser.add_argument("--split",          type=str,      default="val",
                    choices=["train", "val", "all"])
parser.add_argument("--num_classes",    type=int,      default=4)
parser.add_argument("--input_channels", type=int,      default=4)
parser.add_argument("--img_size",       type=int,      default=192)

# Performance
parser.add_argument("--batch_size",     type=int,      default=0,
                    help="Inference batch size (<=0 = auto-detect)")
parser.add_argument("--num_workers",    type=int,      default=-1)
parser.add_argument("--channels_last",  type=str2bool, default=True)
parser.add_argument("--use_tf32",       type=str2bool, default=True)

# Output control
parser.add_argument("--save_visuals",   type=str2bool, default=True,
                    help="Save color prediction masks and overlay PNG images")
parser.add_argument("--save_eval_pairs",type=str2bool, default=True,
                    help="Save GT/pred .npy pairs for evaluate_metrics.py")
parser.add_argument("--save_edge_maps", type=str2bool, default=False,
                    help="Save predicted edge maps (sigmoid of lateral_edge) as PNGs")
parser.add_argument("--output_dir",     type=str,      default=DEFAULT_OUTPUT_DIR)

# Reproducibility
parser.add_argument("--seed",           type=int,      default=1234)
parser.add_argument("--val_split",      type=float,    default=0.1)

args = parser.parse_args()


# =============================================================================
# Checkpoint Auto-Detection
# =============================================================================

def resolve_auto_checkpoint(img_size):
    """
    Search the outputs directory for a trained checkpoint.

    Looks for best_model.pth first, then falls back to last_model.pth.
    Returns the most recently modified checkpoint that matches the image size.
    """
    exp_dir = os.path.join(BASE_DIR, "outputs", "EA_ResNetUNet_v2_brats")

    # Try best_model.pth first
    pattern    = os.path.join(exp_dir, f"EA_v2_epo*_bs*_{img_size}", "best_model.pth")
    candidates = glob.glob(pattern)

    if not candidates:
        # Fallback to last_model.pth
        pattern    = os.path.join(exp_dir, f"EA_v2_epo*_bs*_{img_size}", "last_model.pth")
        candidates = glob.glob(pattern)

    if not candidates:
        return ""

    # Return the most recently modified checkpoint
    return max(candidates, key=os.path.getmtime)


# =============================================================================
# Inference Loop
# =============================================================================

def run_inference(args, model, loader, device):
    """
    Run model inference on all batches in the loader.

    Generates:
        - Segmentation prediction masks (color PNG)
        - Overlay images: FLAIR / GT / Prediction  (PNG)
        - Per-class binary masks (PNG)
        - Edge prediction maps (PNG, optional)
        - GT/pred .npy pairs for metric evaluation

    Args:
        args:   Parsed arguments.
        model:  Loaded ResNetUNet v2 model.
        loader: DataLoader (val/train/all split).
        device: torch.device.

    Returns:
        total_slices: Number of slices processed.
    """
    model.eval()
    total_slices = 0

    # ---- Create output directories ----
    if args.save_visuals:
        pred_dir       = os.path.join(args.output_dir, "predictions")
        overlay_dir    = os.path.join(args.output_dir, "overlays")
        class_mask_dir = os.path.join(args.output_dir, "class_masks")
        os.makedirs(pred_dir,       exist_ok=True)
        os.makedirs(overlay_dir,    exist_ok=True)
        os.makedirs(class_mask_dir, exist_ok=True)

    if args.save_edge_maps:
        # Edge prediction maps: visualizes what boundaries the model has learned
        edge_dir = os.path.join(args.output_dir, "edge_maps")
        os.makedirs(edge_dir, exist_ok=True)

    if args.save_eval_pairs:
        # Always save inside Phase2_Training/evaluation_metrics/
        eval_gt_dir   = os.path.join(BASE_DIR, "evaluation_metrics", "GT")
        eval_pred_dir = os.path.join(BASE_DIR, "evaluation_metrics", "pred")
        os.makedirs(eval_gt_dir,   exist_ok=True)
        os.makedirs(eval_pred_dir, exist_ok=True)
        # Clear old predictions so we only evaluate fresh results
        for old_f in glob.glob(os.path.join(eval_pred_dir, "*.npy")):
            os.remove(old_f)
        for old_f in glob.glob(os.path.join(eval_gt_dir, "*.npy")):
            os.remove(old_f)

    with torch.no_grad():
        for image_batch, label_batch, meta in tqdm(
            loader, total=len(loader), ncols=90, desc="Inference"
        ):
            if args.channels_last and device.type == "cuda":
                image_batch = image_batch.contiguous(memory_format=torch.channels_last)

            image_batch = image_batch.to(device, non_blocking=True)

            # ---- Forward pass ----
            # Both logits and lateral_edge are returned.
            # lateral_edge is used for edge map visualization if requested.
            # At inference, the edge attention routing happens INSIDE the model
            # automatically — no extra work needed here.
            logits, lateral_edge = model(image_batch)

            # ---- Segmentation predictions ----
            # argmax over softmax → most probable class per pixel
            pred_batch = torch.argmax(torch.softmax(logits, dim=1), dim=1)

            # ---- Edge maps (optional) ----
            if args.save_edge_maps:
                # sigmoid converts raw logit → probability in [0, 1]
                # Higher = more likely to be a boundary pixel
                edge_prob = torch.sigmoid(lateral_edge).cpu().numpy()

            # ---- Move to CPU for numpy operations ----
            pred_np  = pred_batch.cpu().numpy()
            label_np = label_batch.numpy()
            image_np = image_batch.cpu().numpy()

            # ---- Per-slice post-processing ----
            for b in range(pred_np.shape[0]):
                pred_slice  = pred_np[b].astype(np.uint8)
                label_slice = label_np[b].astype(np.uint8)
                image_slice = image_np[b]

                case_id     = meta["case_id"][b]
                slice_index = int(meta["slice_index"][b])
                sample_id   = f"{case_id}_slice_{slice_index:03d}"
                total_slices += 1

                # ---- Save prediction visualization ----
                if args.save_visuals:
                    # Color-coded seg mask (PNG)
                    save_segmentation_mask(
                        pred_slice,
                        os.path.join(pred_dir, f"{sample_id}_pred.png"),
                    )

                    # FLAIR background + GT / Prediction overlay (PNG)
                    plot_predictions(
                        image=image_slice,
                        gt=label_slice,
                        pred=pred_slice,
                        output_path=os.path.join(overlay_dir, f"{sample_id}_overlay.png"),
                        class_names=BRATS_CLASS_NAMES[:args.num_classes],
                    )

                    # Individual binary mask per foreground class (PNG)
                    save_class_masks(
                        pred_slice,
                        os.path.join(class_mask_dir, sample_id),
                        prefix=sample_id,
                        class_names=BRATS_CLASS_NAMES[:args.num_classes],
                    )

                # ---- Save edge attention map ----
                if args.save_edge_maps:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    # edge_prob[b, 0] is the per-pixel boundary probability
                    edge_map = edge_prob[b, 0]  # [H, W], values in [0, 1]
                    plt.imsave(
                        os.path.join(edge_dir, f"{sample_id}_edge.png"),
                        edge_map,
                        cmap="hot",
                        vmin=0, vmax=1,
                    )

                # ---- Save GT/pred .npy pairs for evaluation pipeline ----
                if args.save_eval_pairs:
                    np.save(os.path.join(eval_gt_dir,   f"{sample_id}.npy"), label_slice)
                    np.save(os.path.join(eval_pred_dir, f"{sample_id}.npy"), pred_slice)

    return total_slices


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()

    # ---- Reproducibility ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = False
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Auto-detect optimal settings ----
    if args.num_workers < 0:
        args.num_workers = auto_num_workers(max_cap=12)

    if args.batch_size <= 0:
        args.batch_size = auto_inference_batch_size(args.img_size)

    # ---- Resolve checkpoint ----
    if not args.checkpoint:
        args.checkpoint = resolve_auto_checkpoint(args.img_size)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint!r}.\n"
            "Train the model first or provide --checkpoint <path>."
        )

    # ---- GPU optimizations ----
    if device.type == "cuda" and args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.set_float32_matmul_precision("high")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Print configuration ----
    print("=" * 62)
    print("  EA-ResNetUNet v2 — INFERENCE")
    print("=" * 62)
    print(f"  Checkpoint     = {args.checkpoint}")
    print(f"  Dataset root   = {args.root_path}")
    print(f"  Split          = {args.split}")
    print(f"  Image size     = {args.img_size}")
    print(f"  Batch size     = {args.batch_size}")
    print(f"  Num workers    = {args.num_workers}")
    print(f"  Save visuals   = {args.save_visuals}")
    print(f"  Save edge maps = {args.save_edge_maps}")
    print(f"  Save eval pairs= {args.save_eval_pairs}")
    print(f"  Device         = {device}")
    print("=" * 62)

    # ---- Load dataset ----
    dataset = BraTSSliceDataset(
        root_path=args.root_path,
        split=args.split,
        img_size=args.img_size,
        val_split=args.val_split,
        random_state=41,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"  Total slices   = {len(dataset)}")

    # ---- Load model ----
    model = ResNetUNet(
        num_classes=args.num_classes,
        input_channels=args.input_channels,
    ).to(device)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Load checkpoint weights (strict=True by default)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"  Loaded weights from: {args.checkpoint}")

    # ---- Run inference ----
    total_slices = run_inference(args, model, loader, device)

    # ---- Summary ----
    elapsed = time.time() - start_time
    print()
    print("=" * 62)
    print("  INFERENCE COMPLETE")
    print("=" * 62)
    print(f"  Processed {total_slices} slices in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if args.save_visuals:
        print(f"  Visual outputs  → {args.output_dir}/")

    if args.save_edge_maps:
        print(f"  Edge maps       → {args.output_dir}/edge_maps/")

    if args.save_eval_pairs:
        eval_gt_dir   = os.path.join(BASE_DIR, "evaluation_metrics", "GT")
        eval_pred_dir = os.path.join(BASE_DIR, "evaluation_metrics", "pred")
        gt_count      = len([f for f in os.listdir(eval_gt_dir)   if f.endswith(".npy")])
        pred_count    = len([f for f in os.listdir(eval_pred_dir) if f.endswith(".npy")])
        print(f"  GT files        → {eval_gt_dir}/ ({gt_count} files)")
        print(f"  Pred files      → {eval_pred_dir}/ ({pred_count} files)")
        print()
        print("  To compute all metrics including Boundary & Edge metrics:")
        print("    python evaluation_metrics/evaluate_metrics.py")

    print("=" * 62)
