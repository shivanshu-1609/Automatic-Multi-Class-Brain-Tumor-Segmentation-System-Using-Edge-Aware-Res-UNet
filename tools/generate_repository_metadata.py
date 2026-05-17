"""Generate lightweight repository metadata without committing dataset arrays.

The BraTS processed data and evaluation arrays are intentionally excluded from
Git. This script records compact CSV/JSON summaries that make the repository
auditable and reproducible without uploading raw MRI slices, masks, predictions,
or model checkpoints.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

try:
    import numpy as np
except Exception:  # pragma: no cover - metadata can still be generated.
    np = None


REPO_ROOT = Path(__file__).resolve().parents[1]
METADATA_DIR = REPO_ROOT / "metadata"

PROCESSED_ROOT = REPO_ROOT / "processed_brats_2d"
EDGE_CASE_ROOT = REPO_ROOT / "all_155_slices_case_00006"
EVALUATION_ROOTS = [
    REPO_ROOT / "1. Simple_UNet" / "evaluation_metrics",
    REPO_ROOT / "2. Ea+UNet" / "evaluation_metrics",
]

CASE_SLICE_RE = re.compile(r"(?P<case_id>.+)_slice_(?P<slice_index>\d+)$")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def int_value(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def float_value(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def basename_from_any_path(value: str) -> str:
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if normalized else ""


def parse_case_slice(filename: str) -> tuple[str, int | None]:
    stem = Path(filename).stem
    match = CASE_SLICE_RE.match(stem)
    if not match:
        return stem, None
    return match.group("case_id"), int(match.group("slice_index"))


def inspect_npy(path: Path, include_unique: bool = False) -> dict[str, object]:
    if np is None or not path.exists():
        return {}
    array = np.load(path, mmap_mode="r")
    info: dict[str, object] = {
        "path": rel(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if include_unique:
        info["unique_values"] = [int(v) for v in np.unique(array)]
    return info


def directory_file_stats(root: Path) -> dict[str, object]:
    by_extension: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_directory: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "bytes": 0})
    total_count = 0
    total_bytes = 0

    if not root.exists():
        return {
            "exists": False,
            "files": 0,
            "bytes": 0,
            "by_extension": {},
            "by_directory": {},
        }

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        ext = path.suffix.lower() or "<none>"
        total_count += 1
        total_bytes += size
        by_extension[ext]["count"] += 1
        by_extension[ext]["bytes"] += size
        parent = rel(path.parent)
        by_directory[parent]["count"] += 1
        by_directory[parent]["bytes"] += size

    return {
        "exists": True,
        "files": total_count,
        "bytes": total_bytes,
        "by_extension": dict(sorted(by_extension.items())),
        "by_directory": dict(sorted(by_directory.items())),
    }


def generate_processed_metadata() -> None:
    metadata_rows = csv_rows(PROCESSED_ROOT / "metadata.csv")
    summary_rows = csv_rows(PROCESSED_ROOT / "summary.csv")

    sanitized_manifest: list[dict[str, object]] = []
    for row in metadata_rows:
        split = row.get("split", "")
        image_name = basename_from_any_path(row.get("image_path", ""))
        mask_name = basename_from_any_path(row.get("mask_path", ""))
        sanitized_manifest.append(
            {
                "case_id": row.get("case_id", ""),
                "split": split,
                "slice_index": row.get("slice_index", ""),
                "has_tumor": row.get("has_tumor", ""),
                "image_path": f"processed_brats_2d/{split}/images/{image_name}",
                "mask_path": f"processed_brats_2d/{split}/masks/{mask_name}",
            }
        )

    case_rows: list[dict[str, object]] = []
    for row in summary_rows:
        case_rows.append(
            {
                "case_id": row.get("case_id", ""),
                "split": row.get("split", ""),
                "saved_slices": int_value(row.get("saved_slices")),
                "candidate_slices": int_value(row.get("candidate_slices")),
                "top_k_slices": int_value(row.get("top_k_slices")),
                "context_slices_added": int_value(row.get("context_slices_added")),
                "rare_class_slices": int_value(row.get("rare_class_slices")),
                "removed_redundant": int_value(row.get("removed_redundant")),
            }
        )

    write_csv(
        METADATA_DIR / "processed_brats_2d_manifest.csv",
        ["case_id", "split", "slice_index", "has_tumor", "image_path", "mask_path"],
        sanitized_manifest,
    )
    write_csv(
        METADATA_DIR / "processed_brats_2d_cases.csv",
        [
            "case_id",
            "split",
            "saved_slices",
            "candidate_slices",
            "top_k_slices",
            "context_slices_added",
            "rare_class_slices",
            "removed_redundant",
        ],
        case_rows,
    )

    split_counts = Counter(row["split"] for row in sanitized_manifest)
    tumor_counts = Counter(row["has_tumor"] for row in sanitized_manifest)
    cases_by_split: dict[str, set[str]] = defaultdict(set)
    for row in sanitized_manifest:
        cases_by_split[str(row["split"])].add(str(row["case_id"]))

    saved_slices = [int_value(row["saved_slices"]) for row in case_rows]
    candidate_slices = [int_value(row["candidate_slices"]) for row in case_rows]
    removed_redundant = [int_value(row["removed_redundant"]) for row in case_rows]

    image_samples = sorted((PROCESSED_ROOT / "train" / "images").glob("*.npy"))
    mask_samples = sorted((PROCESSED_ROOT / "train" / "masks").glob("*.npy"))
    if not image_samples:
        image_samples = sorted(PROCESSED_ROOT.glob("**/images/*.npy"))
    if not mask_samples:
        mask_samples = sorted(PROCESSED_ROOT.glob("**/masks/*.npy"))

    summary = {
        "generated_at_utc": now_utc(),
        "source_root": rel(PROCESSED_ROOT),
        "data_policy": "Raw and processed .npy arrays are intentionally ignored by Git; this file records metadata only.",
        "file_stats": directory_file_stats(PROCESSED_ROOT),
        "manifest_rows": len(sanitized_manifest),
        "case_rows": len(case_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "tumor_slice_counts": dict(sorted(tumor_counts.items())),
        "cases_by_split": {key: len(value) for key, value in sorted(cases_by_split.items())},
        "saved_slices_total": sum(saved_slices),
        "saved_slices_mean_per_case": round(mean(saved_slices), 4) if saved_slices else 0,
        "candidate_slices_total": sum(candidate_slices),
        "removed_redundant_total": sum(removed_redundant),
        "sample_image": inspect_npy(image_samples[0]) if image_samples else {},
        "sample_mask": inspect_npy(mask_samples[0], include_unique=True) if mask_samples else {},
    }
    write_json(METADATA_DIR / "processed_brats_2d_summary.json", summary)


def summarize_evaluation_folder(root: Path, subdir: str) -> dict[str, object]:
    files = sorted((root / subdir).glob("*.npy"))
    cases: Counter[str] = Counter()
    slices: list[int] = []
    for path in files:
        case_id, slice_index = parse_case_slice(path.name)
        cases[case_id] += 1
        if slice_index is not None:
            slices.append(slice_index)

    return {
        "model": root.parent.name,
        "folder": rel(root / subdir),
        "kind": subdir,
        "file_count": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "case_count": len(cases),
        "min_slice_index": min(slices) if slices else "",
        "max_slice_index": max(slices) if slices else "",
        "sample": inspect_npy(files[0], include_unique=True) if files else {},
    }


def generate_evaluation_metadata() -> None:
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    for root in EVALUATION_ROOTS:
        for subdir in ("GT", "pred"):
            summary = summarize_evaluation_folder(root, subdir)
            details.append(summary)
            rows.append(
                {
                    "model": summary["model"],
                    "kind": summary["kind"],
                    "folder": summary["folder"],
                    "file_count": summary["file_count"],
                    "bytes": summary["bytes"],
                    "case_count": summary["case_count"],
                    "min_slice_index": summary["min_slice_index"],
                    "max_slice_index": summary["max_slice_index"],
                    "sample_shape": summary["sample"].get("shape", ""),
                    "sample_dtype": summary["sample"].get("dtype", ""),
                    "sample_unique_values": summary["sample"].get("unique_values", ""),
                }
            )

    write_csv(
        METADATA_DIR / "evaluation_artifacts_summary.csv",
        [
            "model",
            "kind",
            "folder",
            "file_count",
            "bytes",
            "case_count",
            "min_slice_index",
            "max_slice_index",
            "sample_shape",
            "sample_dtype",
            "sample_unique_values",
        ],
        rows,
    )
    write_json(
        METADATA_DIR / "evaluation_artifacts_summary.json",
        {
            "generated_at_utc": now_utc(),
            "data_policy": "Evaluation GT/pred .npy arrays are excluded from Git; metrics CSV files remain tracked.",
            "artifacts": details,
        },
    )


def generate_training_run_summary() -> None:
    rows: list[dict[str, object]] = []
    for log_path in sorted(REPO_ROOT.glob("**/outputs/**/log.csv")):
        history = csv_rows(log_path)
        if not history:
            continue
        val_metric_name = "dice" if "dice" in history[0] else "val_dice"
        best_row = max(history, key=lambda row: float_value(row.get(val_metric_name)))
        min_loss_row = min(history, key=lambda row: float_value(row.get("val_loss"), float("inf")))
        final_row = history[-1]
        rows.append(
            {
                "run_log": rel(log_path),
                "epochs": len(history),
                "best_val_dice": best_row.get(val_metric_name, ""),
                "best_val_dice_epoch": best_row.get("epoch", ""),
                "min_val_loss": min_loss_row.get("val_loss", ""),
                "min_val_loss_epoch": min_loss_row.get("epoch", ""),
                "final_lr": final_row.get("lr", ""),
                "final_loss": final_row.get("loss", ""),
                "final_train_dice": final_row.get("train_dice", ""),
                "final_val_loss": final_row.get("val_loss", ""),
                "final_val_dice": final_row.get(val_metric_name, ""),
            }
        )

    write_csv(
        METADATA_DIR / "training_runs_summary.csv",
        [
            "run_log",
            "epochs",
            "best_val_dice",
            "best_val_dice_epoch",
            "min_val_loss",
            "min_val_loss_epoch",
            "final_lr",
            "final_loss",
            "final_train_dice",
            "final_val_loss",
            "final_val_dice",
        ],
        rows,
    )


def generate_edge_case_summary() -> None:
    write_json(
        METADATA_DIR / "edge_visualization_case_00006_summary.json",
        {
            "generated_at_utc": now_utc(),
            "source_root": rel(EDGE_CASE_ROOT),
            "data_policy": "Full per-slice edge visualization images are local artifacts; this summary keeps counts and sizes only.",
            "file_stats": directory_file_stats(EDGE_CASE_ROOT),
        },
    )


def main() -> None:
    METADATA_DIR.mkdir(exist_ok=True)
    generate_processed_metadata()
    generate_evaluation_metadata()
    generate_training_run_summary()
    generate_edge_case_summary()


if __name__ == "__main__":
    main()
