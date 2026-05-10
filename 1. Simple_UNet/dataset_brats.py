import glob
import os
import csv
from functools import lru_cache

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split


# BraTS Class Labels: Background (0), Necrotic/Non-Enhancing (1), Edema (2), Enhancing Tumor (3)
BRATS_CLASS_NAMES = ["Background", "NCR/NET", "ED", "ET"]
# Standard MRI Modalities for segmentation
BRATS_MODALITIES = ["t1", "t1ce", "t2", "flair"]

def is_preprocessed_2d_root(root_path):
    """Check if the directory contains a pre-processed 2D slice dataset."""
    return all(
        os.path.isdir(os.path.join(root_path, split, subdir))
        for split in ("train", "val")
        for subdir in ("images", "masks")
    )

def remap_brats_labels(label_volume):
    """Convert original BraTS labels {0, 1, 2, 4} to sequential indices {0, 1, 2, 3}."""
    remapped = np.zeros_like(label_volume, dtype=np.uint8)
    remapped[label_volume == 1] = 1 # NCR/NET
    remapped[label_volume == 2] = 2 # ED
    remapped[label_volume == 4] = 3 # ET
    return remapped

def normalize_modality(volume):
    """Perform Z-score normalization on the MRI volume using non-zero intensities."""
    volume = volume.astype(np.float32)
    mask = volume != 0
    if not np.any(mask): return volume
    
    mean, std = volume[mask].mean(), volume[mask].std()
    normalized = np.zeros_like(volume, dtype=np.float32)
    normalized[mask] = (volume[mask] - mean) / (std + 1e-8)
    return normalized

def discover_brats_cases(root_path):
    """Scan and index valid BraTS cases (groups of modalities + segmentation)."""
    seg_paths = sorted(glob.glob(os.path.join(root_path, "**", "*_seg.nii.gz"), recursive=True))
    if not seg_paths: raise FileNotFoundError(f"No BraTS segmentation files found in {root_path}")

    cases = []
    for seg_path in seg_paths:
        case_dir = os.path.dirname(seg_path)
        case_id = os.path.basename(seg_path).replace("_seg.nii.gz", "")
        
        # Ensure all 4 modalities exist for this case
        mod_paths = {m: os.path.join(case_dir, f"{case_id}_{m}.nii.gz") for m in BRATS_MODALITIES}
        if all(os.path.exists(p) for p in mod_paths.values()):
            cases.append({"case_id": case_id, "case_dir": case_dir, "modalities": mod_paths, "seg": seg_path})
    
    return sorted(cases, key=lambda x: x["case_id"])

class BraTSSliceDataset(torch.utils.data.Dataset):
    """Dataset class supporting both raw 3D NIfTI volumes and pre-processed 2D .npy slices."""
    def __init__(self, root_path, split="train", img_size=224, val_split=0.1, random_state=41):
        self.root_path = os.path.abspath(root_path)
        self.split = split
        self.img_size = img_size
        
        # Auto-detect if we are working with pre-processed 2D data or raw 3D data
        self.mode = "preprocessed" if is_preprocessed_2d_root(self.root_path) else "raw_3d"

        if self.mode == "preprocessed":
            self.samples = self._build_preprocessed_index(split)
        else:
            all_cases = discover_brats_cases(self.root_path)
            if split == "all": selected_cases = all_cases
            else:
                tr, vl = train_test_split(all_cases, test_size=val_split, random_state=random_state)
                selected_cases = tr if split == "train" else vl
            self.case_lookup = {c["case_id"]: c for c in selected_cases}
            self.samples = self._build_slice_index(selected_cases)

        if not self.samples: raise RuntimeError(f"No samples found for split '{split}' in {root_path}")

    def __len__(self):
        return len(self.samples)

    def _build_slice_index(self, cases):
        """Index all valid slices from 3D cases that contain some brain tissue."""
        samples = []
        for case in cases:
            # We only count slices here; actual loading happens in __getitem__
            # For discovery, we peek at one modality
            vol = nib.load(case["modalities"]["flair"]).get_fdata()
            for s_idx in range(vol.shape[-1]):
                if np.count_nonzero(vol[..., s_idx]) > 0: # Skip completely empty black slices
                    samples.append({"case_id": case["case_id"], "slice_index": s_idx})
        return samples

    def _build_preprocessed_index(self, split):
        """Scan directories for pre-processed .npy files."""
        samples = []
        base_dir = os.path.join(self.root_path, split)
        img_dir, msk_dir = os.path.join(base_dir, "images"), os.path.join(base_dir, "masks")
        for img_p in sorted(glob.glob(os.path.join(img_dir, "*.npy"))):
            fname = os.path.basename(img_p)
            msk_p = os.path.join(msk_dir, fname)
            if os.path.exists(msk_p):
                # Parse case_id and slice_index from filename (e.g., BraTS21_00000_slice_075.npy)
                sample_id = fname.replace(".npy", "")
                cid, s_idx = sample_id.rsplit("_slice_", 1) if "_slice_" in sample_id else (sample_id, 0)
                samples.append({"image_path": img_p, "mask_path": msk_p, "case_id": cid, "slice_index": int(s_idx)})
        return samples

    @staticmethod
    @lru_cache(maxsize=8)
    def _load_case_3d(t1, t1ce, t2, flair, seg):
        """Load and normalize full 3D volumes (with caching to avoid redundant disk I/O)."""
        mods = [normalize_modality(nib.load(p).get_fdata()) for p in [t1, t1ce, t2, flair]]
        img_vol = np.stack(mods, axis=0) # [4, H, W, D]
        lbl_vol = remap_brats_labels(nib.load(seg).get_fdata())
        return img_vol, lbl_vol

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if self.mode == "preprocessed":
            img = np.load(sample["image_path"])
            lbl = np.load(sample["mask_path"])
        else:
            case = self.case_lookup[sample["case_id"]]
            img_vol, lbl_vol = self._load_case_3d(case["modalities"]["t1"], case["modalities"]["t1ce"], 
                                                 case["modalities"]["t2"], case["modalities"]["flair"], case["seg"])
            s_idx = sample["slice_index"]
            img_slice, lbl_slice = img_vol[:, :, :, s_idx], lbl_vol[:, :, s_idx]
            
            # Resize 2D slices to uniform model input size
            img = np.stack([cv2.resize(c, (self.img_size, self.img_size)) for c in img_slice], axis=0)
            lbl = cv2.resize(lbl_slice.astype(np.float32), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return torch.from_numpy(img).float(), torch.from_numpy(lbl).long(), {"case_id": sample["case_id"], "slice_index": sample["slice_index"]}