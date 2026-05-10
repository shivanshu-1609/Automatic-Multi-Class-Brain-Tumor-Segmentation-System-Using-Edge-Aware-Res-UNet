
import argparse
import csv
import math
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

from data import (
    BRATS_MODALITIES,
    discover_brats_cases,
    normalize_modality,
    remap_brats_labels,
)
from utils.performance import auto_num_workers


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths
DEFAULT_RAW_ROOT = os.path.join(REPO_ROOT, "BraTS2021_Training_Data")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "processed_brats_2d")


def build_case_split(cases, val_split=0.1, random_state=41):
    """
    Create a patient-wise train/val split to prevent data leakage.
    Each patient's slices go entirely into either train or val set.
    """
    case_ids = [case["case_id"] for case in cases]

    rng = random.Random(random_state)
    shuffled_case_ids = case_ids[:]
    rng.shuffle(shuffled_case_ids)

    val_count = max(1, int(round(len(shuffled_case_ids) * val_split)))
    val_case_ids = set(shuffled_case_ids[:val_count])

    split_map = {}
    for case in cases:
        split_map[case["case_id"]] = "val" if case["case_id"] in val_case_ids else "train"

    return split_map


def load_case(case):
    """Load all 4 MRI modalities and the segmentation mask for a single case."""
    import nibabel as nib

    modalities = []
    for modality in BRATS_MODALITIES:
        volume = nib.load(case["modalities"][modality]).get_fdata().astype(np.float32)
        modalities.append(normalize_modality(volume))

    label_volume = nib.load(case["seg"]).get_fdata().astype(np.int16)
    label_volume = remap_brats_labels(label_volume)

    image_volume = np.stack(modalities, axis=0)  # [4, H, W, D]

    return image_volume, label_volume


def select_slice_indices_atiss(
    label_volume,
    top_k=30,
    context_radius=1,
    area_threshold=0.001,
    similarity_threshold=0.95,
    rare_class_fraction=0.10,
):
    """
    Adaptive Top-K Informative Slice Selection (ATISS).

    Selection pipeline:
        1. Filter slices by minimum tumor area ratio
        2. Score each slice using weighted combination:
           - 0.5 × tumor area coverage
           - 0.3 × class diversity (fraction of foreground classes present)
           - 0.2 × boundary complexity (gradient edge density)
        3. Select top-K highest-scoring slices
        4. Add ±context_radius neighboring slices for spatial continuity
        5. Include slices containing rare classes (class appears in few slices)
        6. Remove redundant consecutive slices using IoU threshold

    Args:
        label_volume:         3D segmentation mask [H, W, D].
        top_k:                Number of top-scoring slices to select.
        context_radius:       Number of neighboring slices to include around each selected slice.
        area_threshold:       Minimum tumor area ratio to consider a slice informative.
        similarity_threshold: IoU threshold for removing redundant adjacent slices.
        rare_class_fraction:  Fraction threshold for defining a class as "rare".

    Returns:
        selected_indices: Sorted list of selected slice indices.
        selection_stats:  Dictionary with selection statistics for logging.
    """
    height, width, depth = label_volume.shape
    total_area = height * width
    max_classes = 3  # BraTS foreground classes: NCR/NET, ED, ET

    valid_slices = []
    slice_stats = []

    # Step 1: Score each slice
    for k in range(depth):
        mask_k = label_volume[:, :, k]
        area = np.sum(mask_k > 0)
        area_ratio = area / total_area

        # Filter non-informative slices (too little tumor content)
        if area_ratio < area_threshold:
            continue

        # Tumor area score
        area_score = area_ratio

        # Class diversity score (number of unique foreground classes / total possible)
        classes_in_slice = np.unique(mask_k)
        classes_in_slice = classes_in_slice[classes_in_slice > 0]
        num_classes = len(classes_in_slice)
        class_score = num_classes / max_classes

        # Boundary complexity score (gradient-based edge density)
        gx, gy = np.gradient(mask_k.astype(np.float32))
        edges = np.abs(gx) + np.abs(gy)
        boundary_pixels = np.sum(edges > 0)
        boundary_score = boundary_pixels / area if area > 0 else 0

        # Weighted composite score
        score_k = 0.5 * area_score + 0.3 * class_score + 0.2 * boundary_score

        valid_slices.append(k)
        slice_stats.append({
            'index': k,
            'score': score_k,
            'classes': set(classes_in_slice),
            'area': area
        })

    # Handle edge case: no informative slices found
    if not valid_slices:
        return [], {
            "candidate_slices": 0,
            "top_k_slices": 0,
            "context_slices_added": 0,
            "rare_class_slices": 0,
            "removed_redundant": 0
        }

    # Step 2: Select top-K by score
    sorted_slices = sorted(slice_stats, key=lambda x: x['score'], reverse=True)
    top_k_indices = [s['index'] for s in sorted_slices[:top_k]]

    # Step 3: Add context slices (neighboring slices for spatial continuity)
    final_slices = set()
    context_slices_added = 0
    for k in top_k_indices:
        for offset in range(-context_radius, context_radius + 1):
            neighbor_k = k + offset
            if 0 <= neighbor_k < depth:
                if neighbor_k not in final_slices:
                    if neighbor_k not in top_k_indices:
                        context_slices_added += 1
                final_slices.add(neighbor_k)

    # Step 4: Preserve slices with rare classes
    all_classes_across_volume = [c for s in slice_stats for c in s['classes']]
    class_frequency = Counter(all_classes_across_volume)
    rare_cutoff = max(1, int(math.ceil(len(valid_slices) * rare_class_fraction)))
    rare_classes = {c for c, count in class_frequency.items() if count <= rare_cutoff}

    rare_class_additions = 0
    for s in slice_stats:
        if s['classes'] & rare_classes:
            if s['index'] not in final_slices:
                final_slices.add(s['index'])
                rare_class_additions += 1

    # Step 5: Remove redundant slices using IoU on consecutive pairs
    final_slices = sorted(list(final_slices))

    non_redundant_slices = []
    if final_slices:
        non_redundant_slices.append(final_slices[0])
        for i in range(1, len(final_slices)):
            k = final_slices[i]
            prev_k = non_redundant_slices[-1]

            mask_k = label_volume[:, :, k]
            mask_prev = label_volume[:, :, prev_k]

            intersection = np.logical_and(mask_k > 0, mask_prev > 0).sum()
            union = np.logical_or(mask_k > 0, mask_prev > 0).sum()
            iou = intersection / union if union > 0 else 1.0

            if iou <= similarity_threshold:
                non_redundant_slices.append(k)

    removed_redundant = len(final_slices) - len(non_redundant_slices)

    selection_stats = {
        "candidate_slices": len(valid_slices),
        "top_k_slices": len(top_k_indices),
        "context_slices_added": context_slices_added,
        "rare_class_slices": rare_class_additions,
        "removed_redundant": removed_redundant
    }

    return non_redundant_slices, selection_stats


def preprocess_case(
    case, split_name, output_root, img_size,
    top_k, context_radius, area_threshold,
    similarity_threshold, rare_class_fraction,
):
    """
    Preprocess a single 3D case: load volume, select slices via ATISS,
    resize, and save as .npy files.
    """
    image_volume, label_volume = load_case(case)

    # Select informative slices using ATISS
    keep_indices, selection_stats = select_slice_indices_atiss(
        label_volume=label_volume,
        top_k=top_k,
        context_radius=context_radius,
        area_threshold=area_threshold,
        similarity_threshold=similarity_threshold,
        rare_class_fraction=rare_class_fraction,
    )

    image_dir = os.path.join(output_root, split_name, "images")
    mask_dir = os.path.join(output_root, split_name, "masks")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rows = []

    for slice_index in keep_indices:
        image_slice = image_volume[..., slice_index]
        label_slice = label_volume[..., slice_index]

        # Resize each modality channel independently (bilinear for images)
        resized_channels = [
            cv2.resize(channel, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            for channel in image_slice
        ]
        resized_image = np.stack(resized_channels, axis=0).astype(np.float32)  # [4, H, W]

        # Nearest-neighbor interpolation for labels to preserve class boundaries
        resized_label = cv2.resize(
            label_slice.astype(np.float32),
            (img_size, img_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

        sample_name = f"{case['case_id']}_slice_{slice_index:03d}.npy"

        image_path = os.path.join(image_dir, sample_name)
        mask_path = os.path.join(mask_dir, sample_name)

        np.save(image_path, resized_image)
        np.save(mask_path, resized_label)

        rows.append(
            {
                "case_id": case["case_id"],
                "split": split_name,
                "slice_index": slice_index,
                "has_tumor": int(np.count_nonzero(label_slice > 0) > 0),
                "image_path": image_path,
                "mask_path": mask_path,
            }
        )

    return rows, selection_stats


def preprocess_case_worker(
    case, split_name, output_root, img_size,
    top_k, context_radius, area_threshold,
    similarity_threshold, rare_class_fraction,
):
    """Top-level worker function for ProcessPoolExecutor"""
    return preprocess_case(
        case=case,
        split_name=split_name,
        output_root=output_root,
        img_size=img_size,
        top_k=top_k,
        context_radius=context_radius,
        area_threshold=area_threshold,
        similarity_threshold=similarity_threshold,
        rare_class_fraction=rare_class_fraction,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS 3D volumes into 2D slices using ATISS."
    )

    # Dataset paths (support both --raw_root and --root_path for convenience)
    parser.add_argument("--raw_root",   type=str, default=DEFAULT_RAW_ROOT,
                        help="BraTS 3D dataset root path")
    parser.add_argument("--root_path",  type=str, default=None,
                        help="Alias for --raw_root (same argument, either works)")
    parser.add_argument("--output_root",type=str, default=DEFAULT_OUTPUT_ROOT,
                        help="Directory where preprocessed 2D .npy slices are saved")
    parser.add_argument("--img_size",   type=int, default=192,
                        help="Output spatial resolution (192 recommended for 6 GB VRAM laptop)")
    parser.add_argument("--val_split",  type=float, default=0.1,
                        help="Fraction of cases reserved for validation")
    parser.add_argument("--random_state",type=int, default=41,
                        help="Random seed for reproducible train/val split")

    # Laptop mode: limit number of cases processed
    parser.add_argument("--max_cases",  type=int, default=500,
                        help="Max BraTS cases to preprocess (500 for laptop mode, 0 or -1 = all)")

    # ATISS parameters
    parser.add_argument("--top_k",      type=int, default=10,
                        help="Top-k informative slices to select per case via ATISS")
    parser.add_argument("--atiss_k",    type=int, default=None,
                        help="Alias for --top_k (same argument, either works)")
    parser.add_argument("--context_radius", type=int, default=1,
                        help="Neighboring slices to add around each selected slice")
    parser.add_argument("--area_threshold", type=float, default=0.001,
                        help="Minimum tumor area ratio for a slice to be eligible")
    parser.add_argument("--similarity_threshold", type=float, default=0.95,
                        help="IoU threshold to prune redundant adjacent slices")
    parser.add_argument("--rare_class_fraction", type=float, default=0.10,
                        help="Fraction threshold to define a tumor class as rare")
    parser.add_argument("--num_workers",type=int, default=1,
                        help="Preprocessing workers (1 = safe on laptop, -1 = auto)")

    args = parser.parse_args()

    # ---- Resolve argument aliases ----
    # --root_path is an alias for --raw_root
    if args.root_path is not None:
        args.raw_root = args.root_path
    # --atiss_k is an alias for --top_k
    if args.atiss_k is not None:
        args.top_k = args.atiss_k
    # Normalize max_cases: 0 or -1 means use all cases
    if args.max_cases <= 0:
        args.max_cases = None

    # ---- Discover cases and apply max_cases limit ----
    cases = discover_brats_cases(args.raw_root)
    print(f"Total cases found      : {len(cases)}")

    if args.max_cases is not None and len(cases) > args.max_cases:
        # Shuffle with fixed seed for reproducibility, then take first N
        rng = random.Random(args.random_state)
        shuffled = cases[:]
        rng.shuffle(shuffled)
        cases = shuffled[:args.max_cases]
        print(f"max_cases={args.max_cases}: using {len(cases)} cases")

    print(f"ATISS top_k            : {args.top_k}")
    print(f"Image size             : {args.img_size}×{args.img_size}")
    print(f"Output root            : {args.output_root}")
    print()

    split_map = build_case_split(cases, val_split=args.val_split, random_state=args.random_state)

    os.makedirs(args.output_root, exist_ok=True)

    metadata_rows = []
    summary_rows = []

    worker_count = args.num_workers
    if worker_count < 0:
        worker_count = auto_num_workers(max_cap=8)

    print(f"ATISS preprocessing workers: {worker_count}")

    if worker_count <= 1:
        # Sequential processing (for debugging or single-core systems)
        for case_index, case in enumerate(cases, start=1):
            split_name = split_map[case["case_id"]]

            case_rows, selection_stats = preprocess_case_worker(
                case=case,
                split_name=split_name,
                output_root=args.output_root,
                img_size=args.img_size,
                top_k=args.top_k,
                context_radius=args.context_radius,
                area_threshold=args.area_threshold,
                similarity_threshold=args.similarity_threshold,
                rare_class_fraction=args.rare_class_fraction,
            )

            metadata_rows.extend(case_rows)
            summary_rows.append(
                {
                    "case_id": case["case_id"],
                    "split": split_name,
                    "saved_slices": len(case_rows),
                    "candidate_slices": selection_stats.get("candidate_slices", 0),
                    "top_k_slices": selection_stats.get("top_k_slices", 0),
                    "context_slices_added": selection_stats.get("context_slices_added", 0),
                    "rare_class_slices": selection_stats.get("rare_class_slices", 0),
                    "removed_redundant": selection_stats.get("removed_redundant", 0),
                }
            )
            print(f"[{case_index}/{len(cases)}] {case['case_id']} -> saved {len(case_rows)} slices")
    else:
        # Parallel processing for faster preprocessing
        future_map = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for case in cases:
                split_name = split_map[case["case_id"]]
                future = executor.submit(
                    preprocess_case_worker,
                    case,
                    split_name,
                    args.output_root,
                    args.img_size,
                    args.top_k,
                    args.context_radius,
                    args.area_threshold,
                    args.similarity_threshold,
                    args.rare_class_fraction,
                )
                future_map[future] = (case["case_id"], split_name)

            completed = 0
            for future in as_completed(future_map):
                case_id, split_name = future_map[future]
                case_rows, selection_stats = future.result()
                completed += 1

                metadata_rows.extend(case_rows)
                summary_rows.append(
                    {
                        "case_id": case_id,
                        "split": split_name,
                        "saved_slices": len(case_rows),
                        "candidate_slices": selection_stats.get("candidate_slices", 0),
                        "top_k_slices": selection_stats.get("top_k_slices", 0),
                        "context_slices_added": selection_stats.get("context_slices_added", 0),
                        "rare_class_slices": selection_stats.get("rare_class_slices", 0),
                        "removed_redundant": selection_stats.get("removed_redundant", 0),
                    }
                )
                print(f"[{completed}/{len(cases)}] {case_id} -> saved {len(case_rows)} slices")

    # Save metadata CSV (per-slice index)
    metadata_csv = os.path.join(args.output_root, "metadata.csv")
    with open(metadata_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["case_id", "split", "slice_index", "has_tumor", "image_path", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save summary CSV (per-case ATISS statistics)
    summary_csv = os.path.join(args.output_root, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_id", "split", "saved_slices", "candidate_slices",
                "top_k_slices", "context_slices_added",
                "rare_class_slices", "removed_redundant",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print dataset statistics
    train_count = sum(1 for row in metadata_rows if row["split"] == "train")
    val_count = sum(1 for row in metadata_rows if row["split"] == "val")

    print(f"\nATISS preprocessing complete.")
    print(f"Processed {len(cases)} cases into 2D slices.")
    print(f"Train slices: {train_count}")
    print(f"Val slices:   {val_count}")
    print(f"Metadata:     {metadata_csv}")


if __name__ == "__main__":
    main()
