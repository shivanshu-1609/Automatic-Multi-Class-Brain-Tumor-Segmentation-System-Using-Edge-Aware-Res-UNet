import os
import sys
import numpy as np
import glob
import random
from scipy.ndimage import distance_transform_edt, binary_erosion

BASE_DIR  = r"d:\Major Project\1. Simple_UNet\evaluation_metrics"

def extract_boundary(binary_mask, thickness=1):
    if not binary_mask.any():
        return np.zeros_like(binary_mask, dtype=bool)
    eroded = binary_erosion(binary_mask, iterations=thickness)
    return binary_mask & ~eroded

def hausdorff_distance_95(gt_mask, pred_mask):
    if not gt_mask.any() and not pred_mask.any():
        return 0.0
    if not gt_mask.any() or not pred_mask.any():
        return np.inf

    gt_border   = extract_boundary(gt_mask)
    pred_border = extract_boundary(pred_mask)

    dt_pred = distance_transform_edt(~pred_border)
    dt_gt   = distance_transform_edt(~gt_border)

    dist_gt_to_pred   = dt_pred[gt_border]
    dist_pred_to_gt   = dt_gt[pred_border]

    all_distances = np.concatenate([dist_gt_to_pred, dist_pred_to_gt])
    return float(np.percentile(all_distances, 95))

def main():
    gt_dir   = os.path.join(BASE_DIR, "GT")
    pred_dir = os.path.join(BASE_DIR, "pred")
    
    gt_files   = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
    
    # randomly select 50 slices
    random.seed(42)
    selected = random.sample(gt_files, 50)
    
    hd95_vals = []
    
    for gt_path in selected:
        filename  = os.path.basename(gt_path)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            continue
            
        gt_slice   = np.load(gt_path)
        pred_slice = np.load(pred_path)
        
        # Calculate for each class and average
        for c in range(1, 4):
            gt_c   = (gt_slice   == c)
            pred_c = (pred_slice == c)
            hd95   = hausdorff_distance_95(gt_c, pred_c)
            if not np.isinf(hd95):
                hd95_vals.append(hd95)
                
    if hd95_vals:
        print(f"Estimated HD95 for Simple UNet (50 random slices): {np.mean(hd95_vals):.4f}")
    else:
        print("No valid HD95 calculated.")

if __name__ == '__main__':
    main()
