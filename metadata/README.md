# Metadata Package

This folder is safe to track in Git. It contains compact metadata and summaries
for local BraTS artifacts while keeping raw data, processed `.npy` arrays,
prediction arrays, checkpoints, and bulk visualizations out of the repository.

Generated files:

- `processed_brats_2d_summary.json`: aggregate counts, split statistics, file
  sizes, and sample array shapes for the local processed BraTS 2D dataset.
- `processed_brats_2d_cases.csv`: per-case slice-selection summary from the
  processed dataset.
- `processed_brats_2d_manifest.csv`: sanitized slice manifest with relative
  paths only. It does not contain image or mask data.
- `evaluation_artifacts_summary.csv` and `.json`: counts and sample metadata for
  local GT/prediction arrays that are intentionally excluded from Git.
- `training_runs_summary.csv`: compact summary of ignored training logs.
- `edge_visualization_case_00006_summary.json`: counts and sizes for the local
  edge-visualization artifact folder.

Regenerate after preprocessing or evaluation:

```bash
python tools/generate_repository_metadata.py
```
