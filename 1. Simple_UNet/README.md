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
   - GT/pred `.npy` arrays are generated locally and ignored by Git; publish `metrics_results.csv` and repository metadata instead.

## Quick run order

```powershell
python preprocess_brats_2d.py
python train_brats.py
python test_brats.py
python evaluation_metrics/evaluate_metrics.py
```

Notes:
- `test_brats.py` (default) writes matching `.npy` GT/pred files into `evaluation_metrics/GT` and `evaluation_metrics/pred`.
- Generated `.npy`, checkpoint, and output folders are intentionally excluded from Git.
- AISS defaults are already configured in `preprocess_brats_2d.py`.
- Speed defaults are auto-tuned for your system class (24 GB NVIDIA GPU + high-core CPU):
  - auto batch size in train/test
  - auto dataloader workers
  - mixed precision + TF32
  - channels-last memory format on CUDA
  - parallel preprocessing workers
- If you want deterministic reproducible runs (slower), use:
  - `python train_brats.py --deterministic 1`
