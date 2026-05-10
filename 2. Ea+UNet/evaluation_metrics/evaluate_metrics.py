"""
evaluate_metrics.py — Phase 2 Comprehensive Evaluation

Metrics computed per-class:
  Standard:  Accuracy, Dice (DSC), Sensitivity, Specificity, IoU, S-Measure, E-Measure, MAE
  Boundary:  Hausdorff Distance 95 (HD95), Boundary Dice (BDice),
             Boundary Precision, Boundary Recall, Boundary F1

Results are printed as a table AND saved to metrics_results.csv in this folder.
"""

import os
import sys
import csv
import numpy as np
import glob
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt, binary_erosion
import cv2

# Add parent directory to path so we can import utils
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, REPO_ROOT)

from utils.metrics import evaluate_multiclass, aggregate_metric_records, METRIC_LABELS

# ─────────────────────────────────────────────────────────────────────────────
# Boundary Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary(binary_mask, thickness=1):
    """Extract boundary pixels from a binary mask using erosion."""
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = binary_erosion(binary_mask, iterations=thickness)
    return binary_mask & ~eroded


def hausdorff_distance_95(gt_mask, pred_mask):
    """
    Compute 95th percentile Hausdorff Distance between two binary masks.
    Lower is better. Returns 0 if both masks are empty, inf if only one is empty.
    """
    if not gt_mask.any() and not pred_mask.any():
        return 0.0
    if not gt_mask.any() or not pred_mask.any():
        return np.inf

    gt_border   = extract_boundary(gt_mask)
    pred_border = extract_boundary(pred_mask)

    # Distance from each GT boundary pixel to nearest predicted boundary pixel
    dt_pred = distance_transform_edt(~pred_border)
    dt_gt   = distance_transform_edt(~gt_border)

    dist_gt_to_pred   = dt_pred[gt_border]
    dist_pred_to_gt   = dt_gt[pred_border]

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
    return float(np.percentile(all_distances, 95))


def boundary_metrics(gt_mask, pred_mask, tolerance=2):
    """
    Compute Boundary Precision, Recall, F1, and Boundary Dice.
    tolerance: allowed pixel distance for a boundary pixel to be considered matched.
    """
    gt_border   = extract_boundary(gt_mask)
    pred_border = extract_boundary(pred_mask)

    if not gt_border.any() and not pred_border.any():
        return {"bdice": 1.0, "bprecision": 1.0, "brecall": 1.0, "bf1": 1.0}
    if not gt_border.any() or not pred_border.any():
        return {"bdice": 0.0, "bprecision": 0.0, "brecall": 0.0, "bf1": 0.0}

    # Dilate boundaries by tolerance to allow near-matches
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    gt_dilated   = cv2.dilate(gt_border.astype(np.uint8),   kernel).astype(bool)
    pred_dilated = cv2.dilate(pred_border.astype(np.uint8), kernel).astype(bool)

    # Precision: fraction of predicted boundary pixels within tolerance of GT boundary
    precision = (pred_border & gt_dilated).sum() / (pred_border.sum() + 1e-8)
    # Recall: fraction of GT boundary pixels within tolerance of predicted boundary
    recall    = (gt_border & pred_dilated).sum() / (gt_border.sum() + 1e-8)

    bf1   = (2 * precision * recall) / (precision + recall + 1e-8)
    bdice = (2 * (pred_border & gt_border).sum()) / (pred_border.sum() + gt_border.sum() + 1e-8)

    return {
        "bdice":       float(bdice),
        "bprecision":  float(precision),
        "brecall":     float(recall),
        "bf1":         float(bf1),
    }


def compute_boundary_metrics_for_slice(gt_slice, pred_slice, num_classes=4):
    """Compute boundary metrics for each foreground class in one 2D slice."""
    class_names = ["Background", "NCR/NET", "ED", "ET"]
    results = {}
    for c in range(1, num_classes):
        gt_c   = (gt_slice   == c)
        pred_c = (pred_slice == c)
        hd95   = hausdorff_distance_95(gt_c, pred_c)
        bm     = boundary_metrics(gt_c, pred_c)
        results[class_names[c]] = {
            "hd95":       hd95,
            "bdice":      bm["bdice"],
            "bprecision": bm["bprecision"],
            "brecall":    bm["brecall"],
            "bf1":        bm["bf1"],
        }
    return results


def aggregate_boundary_records(records):
    """Average boundary metrics across all slices per class."""
    class_names = ["NCR/NET", "ED", "ET"]
    aggregated = {c: {k: [] for k in ["hd95", "bdice", "bprecision", "brecall", "bf1"]}
                  for c in class_names}

    for record in records:
        for c in class_names:
            if c in record:
                for k in ["hd95", "bdice", "bprecision", "brecall", "bf1"]:
                    val = record[c][k]
                    if not np.isinf(val):  # Skip inf HD95 (both masks empty)
                        aggregated[c][k].append(val)

    summary = {}
    for c in class_names:
        summary[c] = {}
        for k in aggregated[c]:
            vals = aggregated[c][k]
            summary[c][k] = float(np.mean(vals)) if vals else 0.0

    # Foreground mean
    summary["Foreground Mean"] = {}
    for k in ["hd95", "bdice", "bprecision", "brecall", "bf1"]:
        vals = [summary[c][k] for c in class_names]
        summary["Foreground Mean"][k] = float(np.mean(vals))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    gt_dir   = os.path.join(BASE_DIR, "GT")
    pred_dir = os.path.join(BASE_DIR, "pred")
    csv_path = os.path.join(BASE_DIR, "metrics_results.csv")

    gt_files   = sorted(glob.glob(os.path.join(gt_dir,   "*.npy")))
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))

    if not gt_files:
        print(f"[ERROR] No ground truth files found in {gt_dir}")
        return

    if len(gt_files) != len(pred_files):
        print(f"[WARNING] Mismatch: GT={len(gt_files)} vs Pred={len(pred_files)} files.")

    print(f"\nFound {len(gt_files)} slices for evaluation.")
    print("Computing Standard + Boundary + Edge metrics...\n")

    standard_records = []
    boundary_records = []

    for gt_path in tqdm(gt_files, desc="Evaluating Slices"):
        filename  = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            continue

        gt_slice   = np.load(gt_path)
        pred_slice = np.load(pred_path)

        # Standard metrics (Dice, IoU, Sensitivity, Specificity, etc.)
        standard_records.append(evaluate_multiclass(gt_slice, pred_slice))

        # Boundary + Edge metrics (HD95, BDice, BPrecision, BRecall, BF1)
        boundary_records.append(compute_boundary_metrics_for_slice(gt_slice, pred_slice))

    # ── Aggregate ──
    print("\nAggregating metrics...")
    std_summary = aggregate_metric_records(standard_records)
    bnd_summary = aggregate_boundary_records(boundary_records)

    # ─────────────── Print Standard Metrics Table ───────────────
    print("\n" + "=" * 110)
    print("  STANDARD SEGMENTATION METRICS")
    print("=" * 110)
    print(f"{'Class':<18}", end="")
    for key in METRIC_LABELS:
        print(f"{METRIC_LABELS[key]:<12}", end="")
    print()
    print("-" * 110)
    for class_name, class_metrics in std_summary["per_class"].items():
        print(f"{class_name:<18}", end="")
        for key in METRIC_LABELS:
            print(f"{class_metrics[key]:<12.4f}", end="")
        print()
    print("-" * 110)
    print(f"{'Foreground Mean':<18}", end="")
    for key in METRIC_LABELS:
        print(f"{std_summary['foreground_mean'][key]:<12.4f}", end="")
    print("\n" + "=" * 110)

    # ─────────────── Print Boundary Metrics Table ───────────────
    bnd_keys   = ["hd95", "bdice", "bprecision", "brecall", "bf1"]
    bnd_labels = ["HD95↓", "BDice↑", "BPrec↑", "BRecall↑", "BF1↑"]
    all_classes = ["NCR/NET", "ED", "ET", "Foreground Mean"]

    print("\n" + "=" * 80)
    print("  BOUNDARY & EDGE METRICS")
    print("=" * 80)
    print(f"{'Class':<18}", end="")
    for lbl in bnd_labels:
        print(f"{lbl:<12}", end="")
    print()
    print("-" * 80)
    for c in all_classes:
        if c in bnd_summary:
            print(f"{c:<18}", end="")
            for k in bnd_keys:
                print(f"{bnd_summary[c][k]:<12.4f}", end="")
            print()
    print("=" * 80)

    # ─────────────── Save to CSV ───────────────
    print(f"\n  Saving results to: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["Class"] + list(METRIC_LABELS.values()) + bnd_labels
        writer.writerow(header)

        # Standard classes (Background + foreground)
        for class_name, class_metrics in std_summary["per_class"].items():
            std_vals = [f"{class_metrics[k]:.4f}" for k in METRIC_LABELS]
            # Boundary values (Background has no boundary metrics)
            if class_name in bnd_summary:
                bnd_vals = [f"{bnd_summary[class_name][k]:.4f}" for k in bnd_keys]
            else:
                bnd_vals = ["N/A"] * len(bnd_keys)
            writer.writerow([class_name] + std_vals + bnd_vals)

        # Foreground means
        std_fg   = [f"{std_summary['foreground_mean'][k]:.4f}" for k in METRIC_LABELS]
        bnd_fg   = [f"{bnd_summary['Foreground Mean'][k]:.4f}" for k in bnd_keys]
        writer.writerow(["Foreground Mean"] + std_fg + bnd_fg)

    print(f"  Done! Open metrics_results.csv to view in Excel.\n")


if __name__ == "__main__":
    main()
