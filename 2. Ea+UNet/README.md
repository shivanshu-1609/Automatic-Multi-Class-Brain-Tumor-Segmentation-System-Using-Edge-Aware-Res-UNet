# Edge-Attention ResNet-UNet v2

This repository contains the code for **Edge-Aware Medical Image Segmentation** on the BraTS dataset, featuring a ResNet-based UNet with an integrated Edge Attention branch.

## 🚀 Deployment
### Recommended: Streamlit Cloud
For the best experience, deploy this repository to **Streamlit Cloud**. It natively supports Streamlit apps and is the easiest way to host this model.

### 3D Volumetric Analysis
- **NIfTI Support**: Upload all 4 MRI modalities (`.nii` or `.nii.gz`).
- **Full Volume Inference**: Processes all 155 slices of a BraTS volume.
- **Tumor Detection**: Automatically identifies which slices contain tumors and highlights them.
- **Interactive Viewer**: Scroll through slices with real-time segmentation overlays and edge attention maps.

### Local Run
```bash
pip install -r requirements.txt
cd Streamlit_App
streamlit run app.py
```

---

## 🏗️ Project Structure
- `Streamlit_App/app.py`: Main Streamlit application.
- `Streamlit_App/best_model.pth`: Local pre-trained model weights. This checkpoint is intentionally ignored by Git.
- `models/`: Architecture implementation (Shared).
- `data/`: Dataset discovery, preprocessing helpers, augmentations, and ATISS slice selection.
- `train_brats.py`: Training entry point.
- `preprocess_brats_2d.py`: Data preparation.
- `requirements.txt`: Combined dependencies for both app and training.

Generated `.npy` data, GT/prediction arrays, checkpoints, and output folders are excluded from Git. Repository-safe summaries are stored in the root `metadata/` folder.

---

## 📈 Training Details
Use `preprocess_brats_2d.py`, `train_brats.py`, `test_brats.py`, and `evaluation_metrics/evaluate_metrics.py` to reproduce the training and evaluation pipeline.

## 🔗 Repository
[https://github.com/shivanshu-1609/Edge-Attention-ResNet-UNet](https://github.com/shivanshu-1609/Edge-Attention-ResNet-UNet)
