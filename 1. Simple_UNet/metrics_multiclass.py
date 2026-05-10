import math
import os
import sys
from collections import OrderedDict

import numpy as np

# Setup path for external metric library (miseval)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MISEVAL_ROOT = os.path.join(REPO_ROOT, "Eval_Metrics", "miseval-master")

if MISEVAL_ROOT not in sys.path:
    sys.path.insert(0, MISEVAL_ROOT)

try:
    from miseval import evaluate

    MISEVAL_AVAILABLE = True
except ModuleNotFoundError:
    # Fallback if miseval dependencies are not installed
    evaluate = None
    MISEVAL_AVAILABLE = False


# BraTS class labels
BRATS_CLASS_NAMES = ["Background", "NCR/NET", "ED", "ET"]

# Supported evaluation metrics
METRIC_LABELS = OrderedDict(
    [
        ("accuracy", "Accuracy"),
        ("dice", "DSC"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("iou", "IoU"),
        ("ssm", "SMeasure"),
        ("em", "EMeasure"),
        ("mae", "MAE"),
    ]
)


def _safe_metric(truth_mask, pred_mask, metric_key):
    # Compute metric with handling for empty masks and fallback logic
    truth_mask = truth_mask.astype(np.uint8)
    pred_mask = pred_mask.astype(np.uint8)

    truth_sum = int(truth_mask.sum())
    pred_sum = int(pred_mask.sum())

    # Handle empty class case
    if truth_sum == 0 and pred_sum == 0:
        return 0.0 if metric_key == "mae" else 1.0

    if not MISEVAL_AVAILABLE:
        return _manual_metric(truth_mask, pred_mask, metric_key)

    try:
        value = evaluate(truth_mask, pred_mask, metric=METRIC_LABELS[metric_key])
    except Exception:
        return _manual_metric(truth_mask, pred_mask, metric_key)

    try:
        value = float(value)
    except TypeError:
        value = float(np.asarray(value).item())

    return float("nan") if math.isinf(value) else value


def _manual_metric(truth_mask, pred_mask, metric_key):
    # Manual implementation of metrics when miseval is unavailable
    truth_mask = truth_mask.astype(bool)
    pred_mask = pred_mask.astype(bool)

    tp = float(np.logical_and(truth_mask, pred_mask).sum())
    tn = float(np.logical_and(~truth_mask, ~pred_mask).sum())
    fp = float(np.logical_and(~truth_mask, pred_mask).sum())
    fn = float(np.logical_and(truth_mask, ~pred_mask).sum())

    if metric_key == "dice":
        denom = 2 * tp + fp + fn
        return 1.0 if denom == 0 else (2 * tp) / denom

    if metric_key == "accuracy":
        total = tp + tn + fp + fn
        return float("nan") if total == 0 else (tp + tn) / total

    if metric_key == "sensitivity":
        denom = tp + fn
        return float("nan") if denom == 0 else tp / denom

    if metric_key == "specificity":
        denom = tn + fp
        return float("nan") if denom == 0 else tn / denom

    if metric_key == "iou":
        denom = tp + fp + fn
        return 1.0 if denom == 0 else tp / denom

    if metric_key == "mae":
        return float(np.abs(pred_mask.astype(np.float32) - truth_mask.astype(np.float32)).mean())

    if metric_key == "em":
        return _eval_e_measure(pred_mask.astype(np.float32), truth_mask.astype(np.float32))

    if metric_key == "ssm":
        return _eval_s_measure(pred_mask.astype(np.float32), truth_mask.astype(np.float32))

    return float("nan")


def _eval_e_measure(pred_mask, truth_mask, num_thresholds=255):
    # Compute E-measure using threshold averaging
    thresholds = np.linspace(0.0, 1.0 - 1e-10, num_thresholds)

    scores = []
    for threshold in thresholds:
        pred_bin = (pred_mask >= threshold).astype(np.float32)

        fm = pred_bin - pred_bin.mean()
        gt = truth_mask.astype(np.float32) - truth_mask.astype(np.float32).mean()

        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) ** 2) / 4

        score = enhanced.sum() / (truth_mask.size - 1 + 1e-20)
        scores.append(score)

    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _eval_s_measure(pred_mask, truth_mask, alpha=0.5):
    # Compute S-measure (structure similarity)
    pred_mask = pred_mask.astype(np.float32)
    truth_mask = truth_mask.astype(np.float32)

    y = truth_mask.mean()

    if y == 0:
        return float(1.0 - pred_mask.mean())
    if y == 1:
        return float(pred_mask.mean())

    gt = truth_mask.copy()
    gt[gt >= 0.5] = 1.0
    gt[gt < 0.5] = 0.0

    q = alpha * _s_object(pred_mask, gt) + (1 - alpha) * _s_region(pred_mask, gt)

    return float(max(q, 0.0))


def _s_object(pred, gt):
    fg = np.where(gt == 0, 0.0, pred)
    bg = np.where(gt == 1, 0.0, 1 - pred)

    o_fg = _object_score(fg, gt)
    o_bg = _object_score(bg, 1 - gt)

    u = gt.mean()

    return u * o_fg + (1 - u) * o_bg


def _object_score(pred, gt):
    temp = pred[gt == 1]

    if temp.size == 0:
        return 0.0

    x = temp.mean()
    sigma_x = temp.std()

    return float((2.0 * x) / (x * x + 1.0 + sigma_x + 1e-20))


def _s_region(pred, gt):
    x, y = _centroid(gt)

    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divide_gt(gt, x, y)
    p1, p2, p3, p4 = _divide_prediction(pred, x, y)

    q1 = _ssim_region(p1, gt1)
    q2 = _ssim_region(p2, gt2)
    q3 = _ssim_region(p3, gt3)
    q4 = _ssim_region(p4, gt4)

    return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4


def _centroid(gt):
    rows, cols = gt.shape[-2:]

    if gt.sum() == 0:
        return int(round(cols / 2)), int(round(rows / 2))

    total = gt.sum()

    i = np.arange(0, cols, dtype=np.float32)
    j = np.arange(0, rows, dtype=np.float32)

    x = int(round((gt.sum(axis=0) * i).sum() / total + 1e-20))
    y = int(round((gt.sum(axis=1) * j).sum() / total + 1e-20))

    return x, y


def _divide_gt(gt, x, y):
    h, w = gt.shape[-2:]
    area = h * w

    lt = gt[:y, :x]
    rt = gt[:y, x:w]
    lb = gt[y:h, :x]
    rb = gt[y:h, x:w]

    x = float(x)
    y = float(y)

    w1 = x * y / area
    w2 = (w - x) * y / area
    w3 = x * (h - y) / area
    w4 = 1 - w1 - w2 - w3

    return lt, rt, lb, rb, w1, w2, w3, w4


def _divide_prediction(pred, x, y):
    h, w = pred.shape[-2:]

    lt = pred[:y, :x]
    rt = pred[:y, x:w]
    lb = pred[y:h, :x]
    rb = pred[y:h, x:w]

    return lt, rt, lb, rb


def _ssim_region(pred, gt):
    gt = gt.astype(np.float32)

    h, w = pred.shape[-2:]
    n = h * w

    if n == 0:
        return 0.0

    x = pred.mean()
    y = gt.mean()

    sigma_x2 = ((pred - x) ** 2).sum() / (n - 1 + 1e-20)
    sigma_y2 = ((gt - y) ** 2).sum() / (n - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-20)

    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        return float(alpha / (beta + 1e-20))

    if alpha == 0 and beta == 0:
        return 1.0

    return 0.0


def evaluate_multiclass(truth, pred, num_classes=4, class_names=None):
    # Compute metrics per class and average over foreground
    class_names = class_names or BRATS_CLASS_NAMES[:num_classes]

    metrics = {"per_class": OrderedDict()}

    for class_index, class_name in enumerate(class_names):
        truth_mask = truth == class_index
        pred_mask = pred == class_index

        metrics["per_class"][class_name] = OrderedDict(
            (metric_key, _safe_metric(truth_mask, pred_mask, metric_key))
            for metric_key in METRIC_LABELS.keys()
        )

    foreground_names = class_names[1:] if len(class_names) > 1 else class_names

    metrics["foreground_mean"] = OrderedDict(
        (
            metric_key,
            float(
                np.nanmean(
                    [metrics["per_class"][class_name][metric_key] for class_name in foreground_names]
                )
            ),
        )
        for metric_key in METRIC_LABELS.keys()
    )

    return metrics


def aggregate_metric_records(records, class_names=None):
    # Aggregate metrics across all samples
    if not records:
        raise ValueError("No metric records were provided.")

    class_names = class_names or list(records[0]["per_class"].keys())

    aggregated = {"per_class": OrderedDict(), "num_samples": len(records)}

    for class_name in class_names:
        aggregated["per_class"][class_name] = OrderedDict()

        for metric_key in METRIC_LABELS.keys():
            values = [record["per_class"][class_name][metric_key] for record in records]
            aggregated["per_class"][class_name][metric_key] = float(np.nanmean(values))

    foreground_names = class_names[1:] if len(class_names) > 1 else class_names

    aggregated["foreground_mean"] = OrderedDict()

    for metric_key in METRIC_LABELS.keys():
        values = [aggregated["per_class"][class_name][metric_key] for class_name in foreground_names]
        aggregated["foreground_mean"][metric_key] = float(np.nanmean(values))

    return aggregated


def summary_to_rows(summary):
    # Convert summary dict into row format
    rows = []

    for class_name, class_metrics in summary["per_class"].items():
        row = {"class": class_name}
        row.update(class_metrics)
        rows.append(row)

    foreground_row = {"class": "Foreground Mean"}
    foreground_row.update(summary["foreground_mean"])
    rows.append(foreground_row)

    return rows


def summary_to_dataframe(summary):
    # Convert summary into pandas DataFrame
    import pandas as pd

    return pd.DataFrame(summary_to_rows(summary))