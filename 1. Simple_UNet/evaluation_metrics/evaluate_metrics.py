import argparse
import csv
import os
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error as sklearn_mae

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
MISEVAL_PATH = REPO_ROOT / "Eval_Metrics" / "miseval-master"

if str(MISEVAL_PATH) not in sys.path:
    sys.path.append(str(MISEVAL_PATH))

try:
    from miseval import evaluate
    HAS_MISEVAL = True
    print(f"INFO: Using miseval from {MISEVAL_PATH}")
except ImportError:
    HAS_MISEVAL = False
    print(f"WARNING: miseval not found at {MISEVAL_PATH}. Falling back to manual implementation.")

# --- BraTS Configuration ---
# BraTS 2021 Class Labels: 0: Background, 1: NCR (Necrotic Tumor), 2: ED (Peritumoral Edema), 3: ET (Enhancing Tumor)
BRATS_CLASS_NAMES = ["Background", "NCR/NET", "ED", "ET"]

# Metrics mapping for the final report
METRICS_MAP = OrderedDict([
    ("accuracy", "Accuracy"),
    ("dice", "Dice (DSC)"),
    ("sensitivity", "Sensitivity"),
    ("specificity", "Specificity"),
    ("iou", "IoU (Jaccard)"),
    ("ssm", "S-Measure"),
    ("em", "E-Measure"),
    ("mae", "MAE")
])

# --- Manual Metric Implementations (SSM, EM) ---


def _eval_e_measure(pred_mask, truth_mask, num_thresholds=255):
    """
    Enhanced-alignment Measure (E-measure)
    Compares the global and local pixel-level alignments between two masks.
    """
    thresholds = np.linspace(0.0, 1.0 - 1e-10, num_thresholds)
    scores = []
    
    # Calculate for multiple thresholds to get an average robust measure
    for threshold in thresholds:
        pred_bin = (pred_mask >= threshold).astype(np.float32)
        
        # Calculate alignment matrix
        fm = pred_bin - pred_bin.mean()
        gt = truth_mask.astype(np.float32) - truth_mask.astype(np.float32).mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        
        # Enhanced alignment score
        enhanced = ((align_matrix + 1) ** 2) / 4
        score = enhanced.sum() / (truth_mask.size - 1 + 1e-20)
        scores.append(score)
        
    return float(np.clip(np.mean(scores), 0.0, 1.0))

def _eval_s_measure(pred_mask, truth_mask, alpha=0.5):
    """
    Structure-based Similarity Measure (S-measure)
    Evaluates regions and objects simultaneously to capture structural similarity.
    Strictly following the guide's implementation methodology.
    """
    pred_mask = pred_mask.astype(np.float32)
    truth_mask = truth_mask.astype(np.float32)
    
    y = truth_mask.mean()
    if y == 0: return float(1.0 - pred_mask.mean())
    if y == 1: return float(pred_mask.mean())
    
    gt = (truth_mask >= 0.5).astype(np.float32)
    
    # Region-based evaluation parts
    def _ssim_region(p, g):
        h, w = p.shape[-2:]
        n = h * w
        if n == 0: return 0.0
        x, y = p.mean(), g.mean()
        sigma_x2 = ((p - x) ** 2).sum() / (n - 1 + 1e-20)
        sigma_y2 = ((g - y) ** 2).sum() / (n - 1 + 1e-20)
        sigma_xy = ((p - x) * (g - y)).sum() / (n - 1 + 1e-20)
        alpha_val = 4 * x * y * sigma_xy
        beta_val = (x * x + y * y) * (sigma_x2 + sigma_y2)
        if alpha_val != 0: return float(alpha_val / (beta_val + 1e-20))
        return 1.0 if alpha_val == 0 and beta_val == 0 else 0.0

    def _s_region(p, g):
        rows, cols = g.shape[-2:]
        total = g.sum()
        if total == 0: x_c, y_c = int(round(cols / 2)), int(round(rows / 2))
        else:
            i, j = np.arange(0, cols, dtype=np.float32), np.arange(0, rows, dtype=np.float32)
            x_c = int(round((g.sum(axis=0) * i).sum() / total + 1e-20))
            y_c = int(round((g.sum(axis=1) * j).sum() / total + 1e-20))
        
        # Divide into parts for structural assessment
        h_g, w_g = g.shape[-2:]
        area = h_g * w_g
        gt_parts = [g[:y_c, :x_c], g[:y_c, x_c:w_g], g[y_c:h_g, :x_c], g[y_c:h_g, x_c:w_g]]
        p_parts = [p[:y_c, :x_c], p[:y_c, x_c:w_g], p[y_c:h_g, :x_c], p[y_c:h_g, x_c:w_g]]
        weights = [float(x_c*y_c)/area, float((w_g-x_c)*y_c)/area, float(x_c*(h_g-y_c))/area]
        weights.append(1.0 - sum(weights))
        return sum(w * _ssim_region(pp, gp) for w, pp, gp in zip(weights, p_parts, gt_parts))

    def _object_score(p, g):
        temp = p[g == 1]
        if temp.size == 0: return 0.0
        x, sigma_x = temp.mean(), temp.std()
        return float((2.0 * x) / (x * x + 1.0 + sigma_x + 1e-20))

    def _s_object(p, g):
        fg, bg = np.where(g == 0, 0.0, p), np.where(g == 1, 0.0, 1 - p)
        u = g.mean()
        return u * _object_score(fg, g) + (1 - u) * _object_score(bg, 1 - g)

    return float(max(alpha * _s_object(pred_mask, gt) + (1 - alpha) * _s_region(pred_mask, gt), 0.0))

# --- Core Evaluation Logic ---

def compute_metrics(truth, pred):
    """
    Compute binary metrics for a single class.
    Uses miseval where available, and falls back to manual implementation.
    """
    # ensure uint8 for miseval
    truth_u8 = truth.astype(np.uint8)
    pred_u8 = pred.astype(np.uint8)
    
    results = {}
    
    if HAS_MISEVAL:
        # Strictly using guide's library for core medical metrics
        # Note: miseval 'evaluate' in binary mode expects class 1 as target
        results["accuracy"] = evaluate(truth_u8, pred_u8, metric="ACC")
        results["dice"] = evaluate(truth_u8, pred_u8, metric="DSC")
        results["sensitivity"] = evaluate(truth_u8, pred_u8, metric="SENS")
        results["specificity"] = evaluate(truth_u8, pred_u8, metric="SPEC")
        results["iou"] = evaluate(truth_u8, pred_u8, metric="Jaccard")
    else:
        # Fallback manual binary metrics
        tp = float(np.logical_and(truth_u8, pred_u8).sum())
        tn = float(np.logical_and(truth_u8==0, pred_u8==0).sum())
        fp = float(np.logical_and(truth_u8==0, pred_u8).sum())
        fn = float(np.logical_and(truth_u8, pred_u8==0).sum())
        results["accuracy"] = (tp + tn) / (tp + tn + fp + fn + 1e-20)
        results["dice"] = (2 * tp) / (2 * tp + fp + fn + 1e-20)
        results["sensitivity"] = tp / (tp + fn + 1e-20)
        results["specificity"] = tn / (tn + fp + 1e-20)
        results["iou"] = tp / (tp + fp + fn + 1e-20)

    # MAE using Scikit-learn as per guide's eval.py
    results["mae"] = sklearn_mae(truth_u8.flatten(), pred_u8.flatten())
    
    # S-Measure and E-Measure 
    results["ssm"] = _eval_s_measure(pred_u8, truth_u8)
    results["em"] = _eval_e_measure(pred_u8, truth_u8)
    
    # Handle perfect empty matches as 1 (standard practice in segmentation metrics)
    if truth_u8.sum() == 0 and pred_u8.sum() == 0:
        for k in ["dice", "sensitivity", "iou", "ssm", "em"]:
            results[k] = 1.0
        results["mae"] = 0.0
            
    return results

def load_mask(path):
    """Load mask from .npy or image files"""
    if str(path).endswith(".npy"):
        return np.load(str(path))
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None: raise FileNotFoundError(f"Mask not found: {path}")
    return mask

def run_evaluation(gt_dir, pred_dir, output_dir, num_classes):
    """
    Main evaluation pipeline.
    This works independently from the testing mechanism by reading saved results.
    """
    print(f"--- Starting Independent Evaluation ---")
    print(f"GT Directory:   {gt_dir}")
    print(f"Pred Directory: {pred_dir}")
    
    # Find matching pairs
    gt_paths = sorted([p for p in Path(gt_dir).iterdir() if p.suffix.lower() in {'.npy', '.png', '.jpg'}])
    pred_paths = {p.stem: p for p in Path(pred_dir).iterdir() if p.suffix.lower() in {'.npy', '.png', '.jpg'}}
    
    valid_pairs = []
    for gp in gt_paths:
        if gp.stem in pred_paths:
            valid_pairs.append((gp, pred_paths[gp.stem]))
            
    if not valid_pairs:
        print("ERROR: No matching GT-Prediction pairs found. Check filenames.")
        return

    print(f"Processing {len(valid_pairs)} samples for {num_classes} classes...")
    
    # Store all per-case results
    all_case_metrics = []
    
    for gt_path, pred_path in valid_pairs:
        gt_mask = load_mask(gt_path)
        pred_mask = load_mask(pred_path)
        
        # Resize if mismatch (though they should match if coming from test_brats.py)
        if gt_mask.shape != pred_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        case_results = {}
        # Multi-class evaluation (One-vs-Rest for each class)
        for c in range(num_classes):
            case_results[c] = compute_metrics(gt_mask == c, pred_mask == c)
            
        all_case_metrics.append(case_results)

    # --- Summarization and Reporting ---
    
    # 1. Average metrics across all cases for each class
    summary_rows = []
    class_means = {}
    
    for c in range(num_classes):
        row = {"Class": BRATS_CLASS_NAMES[c] if c < len(BRATS_CLASS_NAMES) else f"Class_{c}"}
        class_means[c] = {}
        for m_key in METRICS_MAP.keys():
            val = np.mean([case[c][m_key] for case in all_case_metrics])
            row[m_key] = f"{val:.6f}"
            class_means[c][m_key] = val
        summary_rows.append(row)

    # 2. Average metrics for all Foreground classes (1, 2, 3)
    fg_classes = [c for c in range(1, num_classes)]
    if fg_classes:
        fg_row = {"Class": "Foreground Mean"}
        for m_key in METRICS_MAP.keys():
            fg_val = np.mean([class_means[c][m_key] for c in fg_classes])
            fg_row[m_key] = f"{fg_val:.6f}"
        summary_rows.append(fg_row)

    # Save results to CSV
    os.makedirs(output_dir, exist_ok=True)
    report_path = Path(output_dir) / "evaluation_results.csv"
    
    with open(report_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Class"] + list(METRICS_MAP.keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"\nEvaluation Complete!")
    print(f"Full report saved to: {report_path}")
    
    # Display Foreground Summary in Console
    if fg_classes:
        print("\n--- Summary (Foreground Mean) ---")
        for m_key, m_name in METRICS_MAP.items():
            print(f"{m_name:15}: {fg_row[m_key]}")

def main():
    parser = argparse.ArgumentParser(description="Medical Image Segmentation Evaluator")
    parser.add_argument("--gt_dir", type=str, default="GT", help="Path to ground truth masks")
    parser.add_argument("--pred_dir", type=str, default="pred", help="Path to predicted masks")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory for output report")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes in masks")
    args = parser.parse_args()
    
    # Resolve paths relative to the script location
    script_dir = Path(__file__).resolve().parent
    
    gt_path = (script_dir / args.gt_dir).resolve()
    pred_path = (script_dir / args.pred_dir).resolve()
    out_path = (script_dir / args.output_dir).resolve()

    run_evaluation(gt_path, pred_path, out_path, args.num_classes)

if __name__ == "__main__":
    main()
