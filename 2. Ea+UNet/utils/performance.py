"""
performance_profile.py  —  Hardware-aware auto-tuning for RTX 4050 Laptop.

=============================================================================
RTX 4050 LAPTOP GPU PROFILE
=============================================================================

Specs:
    Dedicated VRAM : 6 GB
    Shared memory  : ~7.9 GB (system RAM shared via WDDM)
    Architecture   : Ada Lovelace (Ampere-compatible for TF32)
    CUDA cores     : 2560

Memory budget for our model (ResNet EA-UNet v2, img_size=192):
    Model weights  : ~120 MB
    Batch [8, 4, 192, 192] float16 (AMP): ~9 MB
    Feature maps   : ~600 MB (rough estimate for forward pass)
    Gradient maps  : ~600 MB (backward pass ≈ forward)
    Optimizer state: ~240 MB (SGD: 1× copy of weights)
    Safety buffer  : ~500 MB
    ─────────────────────────────────
    Total at bs=8  : ~2.1 GB  ← safe on 6 GB VRAM

    At bs=12 (192²): ~2.8 GB  ← also feasible, but leaves less headroom
    At bs=16 (192²): ~3.5 GB  ← risky on laptop (VRAM may be stolen by OS/browser)

    At bs=4 (192²) : ~1.2 GB  ← very safe, but slower (fewer GPU utilization)

RECOMMENDATION:
    img_size=192, batch_size=8 → best stability/speed tradeoff on 6 GB Laptop GPU

Why img_size=192 and not 256?
    256×256 quadruples the feature map memory vs 128×128.
    At 256², safe batch would be only 4-6, giving poor gradient estimates.
    192² provides a good balance: enough resolution for tumor boundaries,
    with batch_size=8 for stable training.

=============================================================================
CPU PROFILE (for DataLoader workers)
=============================================================================

System RAM: 15.7 GB total, ~84% used = ~2.5 GB free during training.

Workers should be limited:
    - Each worker loads a 3D NIfTI volume into RAM (LRU cache=4)
    - 1 volume ≈ 4 modalities × (240×240×155) × float32 ≈ 135 MB
    - At 2 workers: ~270 MB RAM usage from workers → safe
    - At 4 workers: ~540 MB → risky with only 2.5 GB free

RECOMMENDATION: num_workers=2 on this laptop.
"""

import os


def get_logical_cpu_count():
    """Return the number of logical CPU cores."""
    return int(os.cpu_count() or 4)


def auto_num_workers(max_cap=12):
    """
    Auto-detect safe DataLoader worker count for this system.

    On the RTX 4050 laptop with ~2.5 GB free RAM:
        - We cap at 2 workers max to avoid OOM from NIfTI volume caching.
        - Each worker caches 4 NIfTI volumes in the LRU cache → ~540 MB per worker.
        - 2 workers = ~270 MB → safe.

    The hard cap of 2 is intentional for laptop use.
    On a server with 64 GB RAM, remove the min(2, ...) constraint.
    """
    logical_cpus = get_logical_cpu_count()
    # On laptop: cap at 2 to protect limited RAM
    # On server: remove the 2 floor/ceiling and use logical_cpus // 2
    return min(2, max(0, min(max_cap, logical_cpus // 2)))


def auto_train_batch_size(img_size=192):
    """
    Auto-detect safe training batch size based on image resolution.

    Tuned for RTX 4050 Laptop (6 GB VRAM) with AMP (float16) enabled.
    AMP roughly halves feature map memory, allowing larger batches.

    Sizes verified against EA-ResNetUNet v2 memory profile:
        img_size <= 128 : bs=16  (2.5 GB estimated)
        img_size <= 192 : bs=8   (2.1 GB estimated)  ← default
        img_size <= 224 : bs=6   (2.8 GB estimated)
        img_size <= 256 : bs=4   (3.0 GB estimated)

    If you get CUDA OOM:
        - Reduce batch_size by 2
        - Or pass --img_size 160 instead of 192

    If you want faster training (GPU utilization > 90%):
        - Try batch_size=12 at img_size=192 (3.0 GB — might push VRAM limits)
    """
    if img_size <= 128:
        return 16
    if img_size <= 192:
        return 8    # recommended for RTX 4050 Laptop at 192×192
    if img_size <= 224:
        return 6
    return 4        # 256×256 — safest for 6 GB VRAM


def auto_inference_batch_size(img_size=192):
    """
    Auto-detect safe inference batch size.

    Inference uses ~50% less VRAM than training (no gradients, no optimizer state).
    So we can use roughly 2× the training batch size at inference.

    These are still conservative for laptop use.
    """
    if img_size <= 128:
        return 32
    if img_size <= 192:
        return 16   # 2× training batch — safe at inference
    if img_size <= 224:
        return 12
    return 8


def get_gpu_info():
    """
    Print GPU information for diagnostic purposes.
    Called at the start of training to confirm hardware is detected correctly.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            print("[HW] CUDA not available — running on CPU")
            return

        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mb  = torch.cuda.get_device_properties(idx).total_memory / 1024**2
        reserved  = torch.cuda.memory_reserved(idx) / 1024**2
        allocated = torch.cuda.memory_allocated(idx) / 1024**2
        free      = total_mb - reserved

        print(f"[HW] GPU  : [{idx}] {name}")
        print(f"[HW] VRAM : {total_mb:.0f} MB total | "
              f"{allocated:.0f} MB allocated | "
              f"{reserved:.0f} MB reserved | "
              f"{free:.0f} MB free")
    except Exception as e:
        print(f"[HW] Could not read GPU info: {e}")
