"""
smoke_test.py — Quick model and dataset sanity check.
Verifies that the model forward pass, ATISS scoring, and augmentation
all work without errors before starting a full training run.
Run: python smoke_test.py
"""
import sys
import traceback
import numpy as np
import torch

print("=" * 60)
print("  EA-ResNetUNet v2 — Smoke Test")
print("=" * 60)

errors = []

# ------------------------------------------------------------------ #
# 1. Model forward pass (no real data needed)
# ------------------------------------------------------------------ #
print("\n[1] Model forward pass...")
try:
    from models import ResNetUNet

    model = ResNetUNet(num_classes=4, input_channels=4)
    model.eval()

    # Fake batch: bs=2, 4 channels, 192×192
    x = torch.zeros(2, 4, 192, 192)
    with torch.no_grad():
        logits, lateral_edge = model(x)

    assert logits.shape       == (2, 4, 192, 192), f"logits shape wrong: {logits.shape}"
    assert lateral_edge.shape == (2, 1, 192, 192), f"edge shape wrong: {lateral_edge.shape}"

    total  = sum(p.numel() for p in model.parameters())
    print("    logits shape      :", tuple(logits.shape))
    print("    lateral_edge shape:", tuple(lateral_edge.shape))
    print("    Total parameters  :", f"{total:,}")
    print("    PASS")
except Exception:
    msg = traceback.format_exc()
    print("    FAIL\n" + msg)
    errors.append("Model forward pass")

# ------------------------------------------------------------------ #
# 2. Edge extraction (GPU-side morphological erosion)
# ------------------------------------------------------------------ #
print("\n[2] Edge extraction from mask...")
try:
    import torch.nn.functional as F

    # Simulate a label batch [B, H, W] with some foreground
    label = torch.zeros(2, 192, 192, dtype=torch.long)
    label[0, 80:112, 80:112] = 1   # NCR/NET square
    label[1, 60:100, 60:120] = 2   # ED rectangle

    from utils.losses import extract_edge_from_mask
    edge = extract_edge_from_mask(label)

    assert edge.shape == (2, 1, 192, 192)
    n_edge_pixels = edge.sum().item()
    assert n_edge_pixels > 0, "No edge pixels found — something is wrong"
    print("    edge shape        :", tuple(edge.shape))
    print("    edge pixel count  :", int(n_edge_pixels))
    print("    PASS")
except Exception:
    msg = traceback.format_exc()
    print("    FAIL\n" + msg)
    errors.append("Edge extraction")

# ------------------------------------------------------------------ #
# 3. ATISS slice scoring
# ------------------------------------------------------------------ #
print("\n[3] ATISS slice scoring...")
try:
    from data.atiss import atiss_score_slice, atiss_select_slices

    # Build a fake label volume [H, W, D=20]
    vol = np.zeros((64, 64, 20), dtype=np.uint8)
    vol[20:44, 20:44, 5:15]  = 1   # NCR/NET in some slices
    vol[10:54, 10:54, 8:12]  = 2   # ED overlapping
    vol[30:40, 30:40, 9:11]  = 3   # ET (small)

    selected = atiss_select_slices(vol, k=5)
    assert len(selected) > 0,  "ATISS returned no slices"
    assert len(selected) <= 5, "ATISS returned more than k slices"
    assert selected == sorted(selected), "ATISS output not sorted"

    # Score a known good and empty slice
    score_tumor = atiss_score_slice(vol[..., 9])   # lots of tumor
    score_empty = atiss_score_slice(vol[..., 0])   # no tumor

    assert score_tumor > score_empty, "ATISS should score tumor slice higher"
    print("    Selected slices   :", selected)
    print("    Tumor slice score :", round(score_tumor, 4))
    print("    Empty slice score :", round(score_empty, 4))
    print("    PASS")
except Exception:
    msg = traceback.format_exc()
    print("    FAIL\n" + msg)
    errors.append("ATISS scoring")

# ------------------------------------------------------------------ #
# 4. Data augmentation pipeline
# ------------------------------------------------------------------ #
print("\n[4] Data augmentation pipeline...")
try:
    from data.augmentations import BraTSAugmentation

    aug   = BraTSAugmentation()
    image = np.random.randn(4, 192, 192).astype(np.float32)
    label = np.random.randint(0, 4, (192, 192)).astype(np.int64)

    aug_image, aug_label = aug(image.copy(), label.copy())

    assert aug_image.shape == (4, 192, 192), f"AUG image shape wrong: {aug_image.shape}"
    assert aug_label.shape == (192, 192),    f"AUG label shape wrong: {aug_label.shape}"
    assert aug_image.dtype == np.float32,    "AUG image dtype should be float32"
    assert aug_label.dtype == np.int64,      "AUG label dtype should be int64"
    assert np.all((aug_label >= 0) & (aug_label < 4)), "AUG introduced invalid class indices"

    print("    Output image shape:", aug_image.shape)
    print("    Output label shape:", aug_label.shape)
    print("    Unique classes     :", sorted(np.unique(aug_label).tolist()))
    print("    PASS")
except Exception:
    msg = traceback.format_exc()
    print("    FAIL\n" + msg)
    errors.append("Data augmentation")

# ------------------------------------------------------------------ #
# 5. Loss functions
# ------------------------------------------------------------------ #
print("\n[5] Loss functions...")
try:
    from utils.losses import BoundaryLoss, DiceLoss, extract_edge_from_mask

    device = torch.device("cpu")

    # Edge loss
    edge_fn   = BoundaryLoss()
    pred_edge = torch.randn(2, 1, 192, 192)
    gt_edge   = torch.randint(0, 2, (2, 1, 192, 192)).float()
    loss_e    = edge_fn(pred_edge, gt_edge)
    assert loss_e.item() > 0, "Edge loss should be positive"

    # Dice loss
    dice_fn   = DiceLoss(num_classes=4)
    logits_4  = torch.randn(2, 4, 192, 192)
    labels    = torch.randint(0, 4, (2, 192, 192))
    loss_d    = dice_fn(logits_4, labels)
    assert loss_d.item() > 0, "Dice loss should be positive"

    print("    BoundaryLoss        :", round(loss_e.item(), 4))
    print("    DiceLoss (4-class):", round(loss_d.item(), 4))
    print("    PASS")
except Exception:
    msg = traceback.format_exc()
    print("    FAIL\n" + msg)
    errors.append("Loss functions")

# ------------------------------------------------------------------ #
# 6. Performance profile
# ------------------------------------------------------------------ #
print("\n[6] Performance profile (hardware tuning)...")
try:
    from utils.performance import (
        auto_num_workers, auto_train_batch_size, auto_inference_batch_size
    )

    workers    = auto_num_workers(max_cap=2)
    train_bs   = auto_train_batch_size(192)
    infer_bs   = auto_inference_batch_size(192)

    assert 0 <= workers <= 2,       "Workers should be 0-2 on laptop"
    assert train_bs == 8,           "Train bs should be 8 at img_size=192 (RTX 4050)"
    assert infer_bs == 16,          "Inference bs should be 16 at img_size=192"

    print("    auto_num_workers()     :", workers)
    print("    auto_train_batch_size(192):", train_bs)
    print("    auto_inference_batch_size(192):", infer_bs)
    print("    PASS")
except Exception:
    msg = traceback.format_exc()
    print("    FAIL\n" + msg)
    errors.append("Performance profile")

# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #
print()
print("=" * 60)
if errors:
    print("  FAILED tests:", ", ".join(errors))
    print("  Fix the above before running train_brats.py")
    sys.exit(1)
else:
    print("  ALL 6 TESTS PASSED")
    print()
    print("  Ready to train! Run:")
    print("    python train_brats.py --root_path D:/MaS-TransUNet/BraTS2021_Training_Data")
print("=" * 60)
