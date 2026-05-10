import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import cv2

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
# Phase2_EdgeVisualization is inside ea+unet_v2 → parent is ea+unet_v2
PROJECT_ROOT   = os.path.dirname(BASE_DIR)

# Add the original ea+unet_v2 project root to Python path
sys.path.insert(0, PROJECT_ROOT)

ORIGINAL_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "outputs", "EA_ResNetUNet_v2_brats",
    "EA_v2_epo50_bs8_cases500_k10_192", "best_model.pth"
)

PHASE2_CHECKPOINT = os.path.join(
    BASE_DIR, "..", "Phase2_Training",
    "outputs", "EA_ResNetUNet_v2_brats",
    "EA_v2_epo50_bs8_cases500_k10_192", "best_model.pth"
)

CASE_DIR = r"D:\MaS-TransUNet\BraTS2021_Training_Data\BraTS2021_00006"
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

from models import ResNetUNet
from utils.losses import extract_edge_from_mask


# ─── Helpers ─────────────────────────────────────────────────────────────────

def normalize_slice(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-8)


def load_model(checkpoint_path, device):
    model = ResNetUNet(num_classes=4, input_channels=4).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def get_edge_prediction(model, image_batch, device):
    with torch.no_grad():
        _, lateral_edge = model(image_batch)
    prob = torch.sigmoid(lateral_edge)[0, 0].cpu().numpy()
    return prob


def colorize_edge(prob_map, threshold=0.3):
    """Convert probability map to a red-highlighted overlay image."""
    norm = np.clip(prob_map / (prob_map.max() + 1e-8), 0, 1)
    colored = plt.cm.hot(norm)[:, :, :3]   # [H, W, 3] RGB
    return colored


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Verify checkpoints exist
    for name, path in [("Original", ORIGINAL_CHECKPOINT), ("Phase2", PHASE2_CHECKPOINT)]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} checkpoint not found at:\n  {path}")
            sys.exit(1)

    print("Loading Original model...")
    model_orig = load_model(ORIGINAL_CHECKPOINT, device)
    print("Loading Phase 2 model...")
    model_ph2  = load_model(PHASE2_CHECKPOINT,   device)
    print("Both models loaded!\n")

    # Load NIfTI volumes
    modalities = ["t1", "t1ce", "t2", "flair"]
    volumes = []
    for mod in modalities:
        path = os.path.join(CASE_DIR, f"BraTS2021_00006_{mod}.nii.gz")
        volumes.append(nib.load(path).get_fdata())
    seg_vol = nib.load(os.path.join(CASE_DIR, "BraTS2021_00006_seg.nii.gz")).get_fdata().astype(np.uint8)
    seg_vol[seg_vol == 4] = 3   # Remap ET label

    saved = 0
    for z in range(155):
        seg_slice = seg_vol[:, :, z]
        if seg_slice.sum() == 0:
            continue    # Skip empty slices

        # Crop to 192×192 (model expected input size)
        s, e = 24, 216
        mod_slices  = [normalize_slice(vol[:, :, z][s:e, s:e]) for vol in volumes]
        seg_cropped = seg_slice[s:e, s:e]

        image_np    = np.stack(mod_slices, axis=0).astype(np.float32)  # [4, 192, 192]
        label_np    = seg_cropped.astype(np.int64)                      # [192, 192]

        image_batch = torch.from_numpy(image_np).unsqueeze(0).to(device)  # [1, 4, 192, 192]
        label_batch = torch.from_numpy(label_np).unsqueeze(0).to(device)  # [1, 192, 192]

        # Canny GT Edge
        edge_gt = extract_edge_from_mask(label_batch, num_classes=4)[0, 0].cpu().numpy()

        # Edge predictions from both models
        prob_orig = get_edge_prediction(model_orig, image_batch, device)
        prob_ph2  = get_edge_prediction(model_ph2,  image_batch, device)

        # Thresholded binary edges
        bin_orig  = (prob_orig > 0.3).astype(np.uint8)
        bin_ph2   = (prob_ph2  > 0.3).astype(np.uint8)

        # Overlay: show GT in green, Phase2 prediction in red, overlap in yellow
        flair = mod_slices[3]  # FLAIR is index 3
        overlay = np.stack([flair, flair, flair], axis=-1)  # [H, W, 3] grayscale
        overlay[edge_gt > 0]  = [0.0, 1.0, 0.0]  # GT boundary → green
        overlay[bin_ph2 > 0]  = [1.0, 0.0, 0.0]  # Phase2 boundary → red
        # Overlap (both correct) → yellow
        both = (edge_gt > 0) & (bin_ph2 > 0)
        overlay[both] = [1.0, 1.0, 0.0]

        # ── 6-panel figure ──────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 6, figsize=(30, 5))
        fig.suptitle(
            f"Edge Comparison & Difference Map — Case 00006, Slice {z:03d}\n"
            f"Difference Map: Red=Phase2 Added | Blue=Phase2 Suppressed",
            fontsize=12, y=1.02
        )

        # Panel 1: FLAIR
        axes[0].imshow(flair, cmap="gray")
        axes[0].set_title("FLAIR Input", fontsize=10)
        axes[0].axis("off")

        # Panel 2: Canny GT
        axes[1].imshow(edge_gt, cmap="gray")
        axes[1].set_title("Canny Edge (GT)", fontsize=10)
        axes[1].axis("off")

        # Panel 3: Original model heatmap (Fixed scale 0-1)
        im3 = axes[2].imshow(prob_orig, cmap="jet", vmin=0, vmax=1.0)
        axes[2].set_title("Original Model\nEdge Heatmap (0-1)", fontsize=10)
        axes[2].axis("off")
        fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Phase 2 model heatmap (Fixed scale 0-1)
        im4 = axes[3].imshow(prob_ph2, cmap="jet", vmin=0, vmax=1.0)
        axes[3].set_title("Phase 2 Model\nEdge Heatmap (0-1)", fontsize=10)
        axes[3].axis("off")
        fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

        # Panel 5: Difference Map (Phase 2 - Original)
        diff = prob_ph2 - prob_orig
        im5 = axes[4].imshow(diff, cmap="bwr", vmin=-0.5, vmax=0.5) # Blue-White-Red
        axes[4].set_title("Difference Map\n(Ph2 - Original)", fontsize=10)
        axes[4].axis("off")
        fig.colorbar(im5, ax=axes[4], fraction=0.046, pad=0.04)

        # Panel 6: Overlay comparison
        axes[5].imshow(overlay)
        axes[5].set_title("Overlay\nGreen=GT | Red=Ph2 | Yellow=Match", fontsize=9)
        axes[5].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"slice_{z:03d}_diff_comparison.png"),
                    dpi=100, bbox_inches="tight")
        plt.close(fig)

        saved += 1
        if saved % 10 == 0:
            print(f"  Processed {saved} slices...")

    print(f"\nDone! Saved {saved} comparison images to:")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
