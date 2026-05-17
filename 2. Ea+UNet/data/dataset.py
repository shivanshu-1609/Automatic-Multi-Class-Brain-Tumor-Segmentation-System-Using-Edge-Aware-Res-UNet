import csv
import glob
import os
import random
from functools import lru_cache

import cv2
import nibabel as nib
import numpy as np
import torch
import torch.utils.data
from sklearn.model_selection import train_test_split

from data.atiss import atiss_select_slices
from data.augmentations import BraTSAugmentation, normalize_modality, remap_brats_labels

BRATS_CLASS_NAMES = ["Background", "NCR/NET", "ED", "ET"]
BRATS_MODALITIES  = ["t1", "t1ce", "t2", "flair"]

def is_preprocessed_2d_root(root_path):
    return all(os.path.isdir(os.path.join(root_path, s, d)) for s in ("train", "val") for d in ("images", "masks"))

def discover_brats_cases(root_path):
    seg_paths = sorted(glob.glob(os.path.join(root_path, "**", "*_seg.nii.gz"), recursive=True))
    if not seg_paths:
        raise FileNotFoundError(f"No BraTS segmentation files found in {root_path!r}.")

    cases = []
    for seg_path in seg_paths:
        case_dir = os.path.dirname(seg_path)
        case_id  = os.path.basename(seg_path).replace("_seg.nii.gz", "")
        modality_paths = {m: os.path.join(case_dir, f"{case_id}_{m}.nii.gz") for m in BRATS_MODALITIES}
        if not all(os.path.exists(p) for p in modality_paths.values()):
            continue
        cases.append({"case_id": case_id, "case_dir": case_dir, "modalities": modality_paths, "seg": seg_path})
    return sorted(cases, key=lambda c: c["case_id"])

class BraTSSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, split="train", img_size=192, val_split=0.1, random_state=41,
                 drop_empty=True, max_cases=None, atiss_k=10, augment=None):
        self.root_path = os.path.abspath(root_path)
        self.split = split
        self.img_size = img_size
        self.val_split = val_split
        self.random_state = random_state
        self.drop_empty = drop_empty
        self.max_cases = max_cases
        self.atiss_k = atiss_k
        self.class_names = BRATS_CLASS_NAMES

        self.augment = (split == "train") if augment is None else augment
        self.augmentor = BraTSAugmentation() if self.augment else None
        self.mode = "preprocessed" if is_preprocessed_2d_root(self.root_path) else "raw_3d"

        if self.mode == "preprocessed":
            self.case_lookup = {}
            self.samples = self._build_preprocessed_index(split)
        else:
            all_cases = discover_brats_cases(self.root_path)
            if self.max_cases is not None and len(all_cases) > self.max_cases:
                rng = random.Random(random_state)
                shuffled = all_cases[:]
                rng.shuffle(shuffled)
                all_cases = shuffled[:self.max_cases]

            if split == "all" or len(all_cases) < 2:
                selected_cases = all_cases
            else:
                train_cases, val_cases = train_test_split(all_cases, test_size=val_split, random_state=random_state, shuffle=True)
                selected_cases = train_cases if split == "train" else val_cases

            self.case_lookup = {c["case_id"]: c for c in selected_cases}
            if atiss_k > 0:
                self.samples = self._build_atiss_slice_index(selected_cases, k=atiss_k)
            else:
                self.samples = self._build_slice_index(selected_cases)

    def __len__(self):
        return len(self.samples)

    def _build_atiss_slice_index(self, cases, k):
        samples = []
        for case in cases:
            try:
                seg_vol = nib.load(case["seg"]).get_fdata().astype(np.int16)
                seg_vol = remap_brats_labels(seg_vol)
            except Exception:
                continue
            selected_indices = atiss_select_slices(seg_vol, k=k)
            for z in selected_indices:
                samples.append({"case_id": case["case_id"], "slice_index": z})
        return samples

    def _build_slice_index(self, cases):
        samples = []
        for case in cases:
            vol = nib.load(case["modalities"]["t1"]).get_fdata()
            depth = vol.shape[-1]
            for z in range(depth):
                if self.drop_empty and np.count_nonzero(vol[..., z]) == 0:
                    continue
                samples.append({"case_id": case["case_id"], "slice_index": z})
        return samples

    def _resolve_data_path(self, stored_path, split_name, subfolder):
        if os.path.exists(stored_path):
            return stored_path
        filename = os.path.basename(stored_path)
        return os.path.join(self.root_path, split_name, subfolder, filename)

    def _build_preprocessed_index(self, split):
        metadata_path = os.path.join(self.root_path, "metadata.csv")
        split_names   = ["train", "val"] if split == "all" else [split]
        samples       = []

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    split_name = row.get("split")
                    if split_name not in split_names:
                        continue
                    image_path = self._resolve_data_path(row.get("image_path", ""), split_name, "images")
                    mask_path = self._resolve_data_path(row.get("mask_path", ""), split_name, "masks")
                    if not image_path or not mask_path:
                        continue
                    try:
                        slice_index = int(row.get("slice_index", -1))
                    except ValueError:
                        slice_index = -1
                    samples.append({"image_path": image_path, "mask_path": mask_path, "case_id": row.get("case_id", ""), "slice_index": slice_index, "split": split_name})
            return samples

        for split_name in split_names:
            image_dir   = os.path.join(self.root_path, split_name, "images")
            mask_dir    = os.path.join(self.root_path, split_name, "masks")
            image_paths = sorted(glob.glob(os.path.join(image_dir, "*.npy")))
            for image_path in image_paths:
                fname     = os.path.basename(image_path)
                mask_path = os.path.join(mask_dir, fname)
                if not os.path.exists(mask_path):
                    continue
                sample_id = os.path.splitext(fname)[0]
                if "_slice_" in sample_id:
                    case_id, slice_token = sample_id.rsplit("_slice_", 1)
                    try: slice_index = int(slice_token)
                    except ValueError: slice_index = -1
                else:
                    case_id, slice_index = sample_id, -1
                samples.append({"image_path": image_path, "mask_path": mask_path, "case_id": case_id, "slice_index": slice_index, "split": split_name})
        return samples

    @staticmethod
    @lru_cache(maxsize=4)
    def _load_case_arrays(t1_path, t1ce_path, t2_path, flair_path, seg_path):
        modalities = []
        for path in (t1_path, t1ce_path, t2_path, flair_path):
            vol = nib.load(path).get_fdata().astype(np.float32)
            modalities.append(normalize_modality(vol))
        label_vol = nib.load(seg_path).get_fdata().astype(np.int16)
        label_vol = remap_brats_labels(label_vol)
        return np.stack(modalities, axis=0), label_vol

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.mode == "preprocessed":
            image = torch.from_numpy(np.load(sample["image_path"])).float()
            label = torch.from_numpy(np.load(sample["mask_path"])).long()
            meta  = {"case_id": sample["case_id"], "slice_index": sample["slice_index"]}
            if self.augment and self.augmentor is not None:
                image_np, label_np = self.augmentor(image.numpy(), label.numpy())
                image = torch.from_numpy(image_np.copy()).float()
                label = torch.from_numpy(label_np.copy()).long()
            return image, label, meta

        case = self.case_lookup[sample["case_id"]]
        slice_index = sample["slice_index"]

        image_vol, label_vol = self._load_case_arrays(
            case["modalities"]["t1"], case["modalities"]["t1ce"],
            case["modalities"]["t2"], case["modalities"]["flair"], case["seg"]
        )

        image_slice = image_vol[..., slice_index]
        label_slice = label_vol[..., slice_index]

        resized_channels = [cv2.resize(image_slice[c], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR) for c in range(image_slice.shape[0])]
        image_np = np.stack(resized_channels, axis=0).astype(np.float32)
        label_np = cv2.resize(label_slice.astype(np.float32), (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST).astype(np.int64)

        if self.augment and self.augmentor is not None:
            image_np, label_np = self.augmentor(image_np, label_np)

        return torch.from_numpy(image_np.copy()).float(), torch.from_numpy(label_np.copy()).long(), {"case_id": case["case_id"], "slice_index": slice_index}
