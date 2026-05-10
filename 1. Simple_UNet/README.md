# Modified_Unet_AISS

This folder is a clean copy of your `Modified_UNet` pipeline with:

1. AISS slice-selection in preprocessing:
   - keep tumor slices (`mask_sum > 0`)
   - add `+/-1` context
   - preserve rare-class and multi-class slices
   - remove tiny-area and redundant neighbors

2. Guide-style evaluation folder:
   - `evaluation_metrics/GT`
   - `evaluation_metrics/pred`
   - `evaluation_metrics/evaluate_metrics.py` (imports `miseval` from `Eval_Metrics/miseval-master`)

## Quick run order

```powershell
python preprocess_brats_2d.py
python train_brats.py
python test_brats.py
python evaluation_metrics/evaluate_metrics.py
```

## One-command fast run (recommended on your system)

```powershell
powershell -ExecutionPolicy Bypass -File .\run_fast_pipeline.ps1
```

This script now uses conda env `brainseg` by default.

Notes:
- `test_brats.py` (default) writes matching `.npy` GT/pred files into `evaluation_metrics/GT` and `evaluation_metrics/pred`.
- AISS defaults are already configured in `preprocess_brats_2d.py`.
- Speed defaults are auto-tuned for your system class (24 GB NVIDIA GPU + high-core CPU):
  - auto batch size in train/test
  - auto dataloader workers
  - mixed precision + TF32
  - channels-last memory format on CUDA
  - parallel preprocessing workers
- If you want deterministic reproducible runs (slower), use:
  - `python train_brats.py --deterministic 1`
