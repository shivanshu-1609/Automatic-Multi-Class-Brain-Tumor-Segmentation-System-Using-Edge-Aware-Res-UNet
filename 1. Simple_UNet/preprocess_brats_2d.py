import argparse
import csv
import math
import os
import random #dataset splitting
from collections import Counter #count class frequencies
from concurrent.futures import ProcessPoolExecutor, as_completed #parallel processing

import cv2
import numpy as np

from dataset_brats import (
    BRATS_MODALITIES,
    discover_brats_cases,
    normalize_modality,
    remap_brats_labels,
)

# Decides how many CPU cores to use
from performance_profile import auto_num_workers


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Dataset paths
DEFAULT_RAW_ROOT = os.path.join(REPO_ROOT, "BraTS2021_Training_Data")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE_DIR, "processed_brats_2d")


def build_case_split(cases, val_split=0.1, random_state=41):
    # Create patient-wise train/val split to avoid data leakage

    # Extract all patient IDs
    case_ids = [case["case_id"] for case in cases]

    # Fix randomness (reproducibility)
    rng = random.Random(random_state)
    shuffled_case_ids = case_ids[:]
    # Shuffle patients
    rng.shuffle(shuffled_case_ids)

    # Decide how many go to validation
    val_count = max(1, int(round(len(shuffled_case_ids) * val_split)))
    val_case_ids = set(shuffled_case_ids[:val_count])

    split_map = {}
    # Assign each case
    for case in cases:
        split_map[case["case_id"]] = "val" if case["case_id"] in val_case_ids else "train"

    return split_map


def load_case(case):
    # Load all MRI modalities and corresponding segmentation mask
    modalities = []

    for modality in BRATS_MODALITIES:
        import nibabel as nib

        volume = nib.load(case["modalities"][modality]).get_fdata().astype(np.float32)
        modalities.append(normalize_modality(volume))

    import nibabel as nib

    label_volume = nib.load(case["seg"]).get_fdata().astype(np.int16)
    label_volume = remap_brats_labels(label_volume)

    image_volume = np.stack(modalities, axis=0)

    return image_volume, label_volume


def select_slice_indices_atiss(
    label_volume,
    top_k=10,
    context_radius=1,
    area_threshold=0.001,
    similarity_threshold=0.95,
    rare_class_fraction=0.10,
):
    """
    Adaptive Top-K Informative Slice Selection (ATISS)
    """
    height, width, depth = label_volume.shape
    total_area = height * width
    max_classes = 3  # BraTS typically has 3 foreground classes: ET, ED, NCR
    
    valid_slices = []
    slice_stats = []

    # Step 1: Iterate over slices
    for k in range(depth):
        mask_k = label_volume[:, :, k]
        area = np.sum(mask_k > 0)
        area_ratio = area / total_area
        
        # Step 2: Filter non-informative slices
        if area_ratio < area_threshold:
            continue
            
        # Step 3: Compute slice features
        # 1. Tumor Area Score
        area_score = area_ratio
        
        # 2. Class Diversity Score
        classes_in_slice = np.unique(mask_k)
        classes_in_slice = classes_in_slice[classes_in_slice > 0] # Ignore background
        num_classes = len(classes_in_slice)
        class_score = num_classes / max_classes
        
        # 3. Boundary Complexity Score using gradient edges
        
        gx, gy = np.gradient(mask_k.astype(np.float32)) 
        # np.gradient() computes rate of change (derivative) of the image
        edges = np.abs(gx) + np.abs(gy)
        boundary_pixels = np.sum(edges > 0)
        boundary_score = boundary_pixels / area if area > 0 else 0
        
        #  Step 4: Compute final score (Weights as recommended: 0.5, 0.3, 0.2)
        score_k = 0.5 * area_score + 0.3 * class_score + 0.2 * boundary_score
        
        valid_slices.append(k)
        slice_stats.append({
            'index': k,
            'score': score_k,
            'classes': set(classes_in_slice),
            'area': area
        })

    # Defensive check: if no valid slice is found
    if not valid_slices:
        return [], {
            "candidate_slices": 0,
            "top_k_slices": 0,
            "context_slices_added": 0,
            "rare_class_slices": 0,
            "removed_redundant": 0
        }

    # Step 5: Select Top-K slices
    # Sort valid slices by score descending
    sorted_slices = sorted(slice_stats, key=lambda x: x['score'], reverse=True)
    top_k_indices = [s['index'] for s in sorted_slices[:top_k]]
    
    # Step 6: Add context slices
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
                
    # Step 7: Ensure rare class preservation
    # Identify rare classes across the whole volume's valid slices
    all_classes_across_volume = [c for s in slice_stats for c in s['classes']]
    class_frequency = Counter(all_classes_across_volume)
    # math.ceil() Round UP to the nearest integer
    rare_cutoff = max(1, int(math.ceil(len(valid_slices) * rare_class_fraction)))
    rare_classes = {c for c, count in class_frequency.items() if count <= rare_cutoff}
    
    rare_class_additions = 0
    for s in slice_stats:
        if s['classes'] & rare_classes:
            if s['index'] not in final_slices:
                final_slices.add(s['index'])
                rare_class_additions += 1
                
    # Step 8: Remove redundant slices using IoU
    final_slices = sorted(list(final_slices))
    
    non_redundant_slices = []
    if final_slices:
        non_redundant_slices.append(final_slices[0])
        for i in range(1, len(final_slices)):
            k = final_slices[i]
            prev_k = non_redundant_slices[-1]
            
            mask_k = label_volume[:, :, k]
            mask_prev = label_volume[:, :, prev_k]
            
            # IoU computation
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
    case,
    split_name,
    output_root,
    img_size,
    top_k,
    context_radius,
    area_threshold,
    similarity_threshold,
    rare_class_fraction,
):
    # Convert 3D volume into filtered 2D slices
    image_volume, label_volume = load_case(case)

    # Use ATISS to select slices
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

        # Resize each modality channel
        resized_channels = [
            cv2.resize(channel, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            for channel in image_slice
        ]

        # Data Conversion Rule: Modalities stacked as channels -> (4, H, W)
        resized_image = np.stack(resized_channels, axis=0).astype(np.float32)

        # Resize segmentation mask
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

        # Store metadata for each slice
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
    case,
    split_name,
    output_root,
    img_size,
    top_k,
    context_radius,
    area_threshold,
    similarity_threshold,
    rare_class_fraction,
):
    # Top-level worker function for ProcessPoolExecutor (Windows spawn-safe).
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
    parser = argparse.ArgumentParser(description="Preprocess BraTS data using ATISS")

    parser.add_argument("--raw_root", type=str, default=DEFAULT_RAW_ROOT, help="BraTS 3D dataset path")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="output 2D dataset path")
    parser.add_argument("--img_size", type=int, default=256, help="image size")
    parser.add_argument("--val_split", type=float, default=0.1, help="validation split")
    parser.add_argument("--random_state", type=int, default=41, help="split seed")
    
    # ATISS Specific Arguments
    # We restrict top_k to 10 (instead of typical 155 slices) to drastically minimize the dataset footprint.
    # This guarantees ONLY the most critical tumor core slices are outputted, 
    # slashing multi-minute epoch training times down to bare minimums.
    parser.add_argument("--top_k", type=int, default=10, help="Number of key slices to select per case (ATISS)")
    parser.add_argument("--context_radius", type=int, default=1, help="ATISS context radius (+/- 1 recommended)")
    parser.add_argument("--area_threshold", type=float, default=0.001, help="Minimum tumor area ratio to consider informative")
    parser.add_argument("--similarity_threshold", type=float, default=0.95, help="IoU threshold to remove redundant slices")
    parser.add_argument("--rare_class_fraction",type=float,default=0.10,help="class is rare if present in <= ceil(positive_slices * rare_class_fraction)",)
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="preprocessing workers (-1 means auto for this system)",
    )

    args = parser.parse_args()

    # Discover all cases and split dataset
    cases = discover_brats_cases(args.raw_root)
    split_map = build_case_split(cases, val_split=args.val_split, random_state=args.random_state)

    os.makedirs(args.output_root, exist_ok=True)

    metadata_rows = []
    summary_rows = []

    worker_count = args.num_workers
    if worker_count < 0:
        worker_count = auto_num_workers(max_cap=8)

    print(f"ATISS preprocessing workers: {worker_count}")

    if worker_count <= 1:
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

    # Save metadata CSV
    metadata_csv = os.path.join(args.output_root, "metadata.csv")

    with open(metadata_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["case_id", "split", "slice_index", "has_tumor", "image_path", "mask_path"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    # Save summary CSV
    summary_csv = os.path.join(args.output_root, "summary.csv")

    with open(summary_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "case_id",
                "split",
                "saved_slices",
                "candidate_slices",
                "top_k_slices",
                "context_slices_added",
                "rare_class_slices",
                "removed_redundant",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print dataset statistics
    train_count = sum(1 for row in metadata_rows if row["split"] == "train")
    val_count = sum(1 for row in metadata_rows if row["split"] == "val")

    print("Processed using ATISS (Adaptive Top-K Informative Slice Selection).")
    print(f"Processed {len(cases)} cases into 2D slices.")
    print(f"Train slices: {train_count}")
    print(f"Val slices: {val_count}")
    print(f"Metadata saved to: {metadata_csv}")


if __name__ == "__main__":
    main()
