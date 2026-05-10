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
streamlit run app.py
```

---

## 🏗️ Project Structure
- `app.py`: Main Streamlit application.
- `best_model.pth`: Pre-trained model weights.
- `models/`: Architecture implementation (Shared).
- `training/`: Full training pipeline, preprocessing scripts, and evaluation metrics.
  - `training/train_brats.py`: Training entry point.
  - `training/preprocess_brats_2d.py`: Data preparation.
- `requirements.txt`: Combined dependencies for both app and training.

---

## 📈 Training Details
For detailed instructions on how to train the model from scratch, please refer to the documentation in the `training/` directory.

## 🔗 Repository
[https://github.com/shivanshu-1609/Edge-Attention-ResNet-UNet](https://github.com/shivanshu-1609/Edge-Attention-ResNet-UNet)
