import numpy as np
import math
from collections import Counter

# Maximum foreground classes (NCR/NET=1, ED=2, ET=3)
_NUM_FG_CLASSES = 3

def atiss_score_slice(label_slice):
    """
    Compute ATISS informativeness score for a single 2D label slice.
    Score = 0.6 × fg_fraction + 0.4 × class_diversity
    """
    total_pixels = label_slice.size
    if total_pixels == 0:
        return 0.0

    fg_pixels   = np.count_nonzero(label_slice)
    fg_fraction = fg_pixels / total_pixels

    unique_fg_classes  = len(np.unique(label_slice[label_slice > 0]))
    class_diversity    = unique_fg_classes / _NUM_FG_CLASSES

    return 0.6 * fg_fraction + 0.4 * class_diversity


def atiss_select_slices(label_volume, k=10):
    """
    Select the k most informative slices from a 3D label volume using basic ATISS scoring.
    """
    depth  = label_volume.shape[-1]
    scores = []

    for z in range(depth):
        score = atiss_score_slice(label_volume[..., z])
        if score > 0:
            scores.append((z, score))

    if not scores:
        return []

    scores.sort(key=lambda item: item[1], reverse=True)
    selected_indices = [z for z, _ in scores[:k]]
    return sorted(selected_indices)


def select_slice_indices_atiss(
    label_volume,
    top_k=30,
    context_radius=1,
    area_threshold=0.001,
    similarity_threshold=0.95,
    rare_class_fraction=0.10,
):
    """
    Advanced Adaptive Top-K Informative Slice Selection (used mainly in preprocessing).
    Selection pipeline:
        1. Filter by area
        2. Score using area + diversity + boundary complexity
        3. Top K
        4. Spatial context
        5. Rare classes
        6. Redundancy filter
    """
    height, width, depth = label_volume.shape
    total_area = height * width
    max_classes = 3

    valid_slices = []
    slice_stats = []

    for k in range(depth):
        mask_k = label_volume[:, :, k]
        area = np.sum(mask_k > 0)
        area_ratio = area / total_area

        if area_ratio < area_threshold:
            continue

        area_score = area_ratio
        classes_in_slice = np.unique(mask_k)
        classes_in_slice = classes_in_slice[classes_in_slice > 0]
        num_classes = len(classes_in_slice)
        class_score = num_classes / max_classes

        gx, gy = np.gradient(mask_k.astype(np.float32))
        edges = np.abs(gx) + np.abs(gy)
        boundary_pixels = np.sum(edges > 0)
        boundary_score = boundary_pixels / area if area > 0 else 0

        score_k = 0.5 * area_score + 0.3 * class_score + 0.2 * boundary_score

        valid_slices.append(k)
        slice_stats.append({
            'index': k,
            'score': score_k,
            'classes': set(classes_in_slice),
            'area': area
        })

    if not valid_slices:
        return [], {
            "candidate_slices": 0,
            "top_k_slices": 0,
            "context_slices_added": 0,
            "rare_class_slices": 0,
            "removed_redundant": 0
        }

    sorted_slices = sorted(slice_stats, key=lambda x: x['score'], reverse=True)
    top_k_indices = [s['index'] for s in sorted_slices[:top_k]]

    final_slices = set()
    context_slices_added = 0
    for k in top_k_indices:
        for offset in range(-context_radius, context_radius + 1):
            neighbor_k = k + offset
            if 0 <= neighbor_k < depth:
                if neighbor_k not in final_slices and neighbor_k not in top_k_indices:
                    context_slices_added += 1
                final_slices.add(neighbor_k)

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
