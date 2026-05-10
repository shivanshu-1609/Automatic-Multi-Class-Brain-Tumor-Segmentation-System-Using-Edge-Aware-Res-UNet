# Automatic Multi-Class Brain Tumor Segmentation System Using Edge Aware Res-UNet

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://edge-attention-resnet-unet.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the official implementation of the **Automatic Multi-Class Brain Tumor Segmentation System Using Edge Aware Res-UNet (Ea+UNet)**. The project addresses the critical challenges in medical image segmentation, specifically class imbalance and fuzzy glioblastoma boundaries, by employing an advanced edge-aware supervision mechanism.

## 📌 Project Overview

Brain tumor segmentation is a vital step in medical diagnostics. This project aims to accurately segment brain tumors from MRI scans into multiple sub-regions (e.g., necrosis, edema, enhancing tumor). We provide a comparative study and implementation of two architectures:

1.  **Simple UNet (Baseline)**: A classic UNet architecture modified for robust performance on the BraTS dataset.
2.  **Edge Aware Res-UNet (Ea+UNet)**: The proposed state-of-the-art model that integrates Residual blocks and an Edge Attention mechanism with boundary-based optimization to capture fine-grained details and fuzzy boundaries, significantly reducing false negatives.

## 📂 Repository Structure

*   `1. Simple_UNet/`: Contains the baseline Modified UNet architecture, training scripts, and evaluation metrics.
*   `2. Ea+UNet/`: Contains the proposed Edge-Aware ResNet-UNet implementation, loss function derivations, and the Streamlit web application for deployment.
*   `Comparison Result/`: Jupyter notebooks and scripts used to compare the performance metrics (e.g., Dice Score) between the Simple UNet and Ea+UNet.

## ✨ Key Features

*   **Advanced Architecture**: Combines ResNet's feature extraction power with UNet's spatial reconstruction.
*   **Edge-Aware Supervision**: Utilizes Boundary Loss to force the network to focus on hard-to-segment edges, improving boundary delineation.
*   **Multi-Class Segmentation**: Supports segmentation of multiple tumor substructures simultaneously.
*   **Streamlit Web App**: Includes a user-friendly web interface (`2. Ea+UNet/Streamlit_App/`) for real-time model inference and visualization.
*   **Comprehensive Evaluation**: Built-in scripts for comprehensive metric calculation (Dice, IoU, Sensitivity, Specificity).

## 🚀 Getting Started

### Prerequisites

*   Python 3.8+
*   PyTorch (CUDA recommended for training)

### Installation

Clone the repository:
```bash
git clone https://github.com/shivanshu-1609/Automatic-Multi-Class-Brain-Tumor-Segmentation-System-Using-Edge-Aware-Res-UNet.git
cd Automatic-Multi-Class-Brain-Tumor-Segmentation-System-Using-Edge-Aware-Res-UNet
```

Install the required dependencies (you can navigate to either `1. Simple_UNet` or `2. Ea+UNet`):
```bash
pip install -r "2. Ea+UNet/requirements.txt"
```

### Dataset

This project utilizes the **BraTS (Brain Tumor Segmentation)** dataset. Please download the dataset and place it in the appropriate `data/` directory or run the provided preprocessing scripts (`preprocess_brats_2d.py`).

## 🖥️ Running the Application

**🌐 Live Demo:** You can try out the deployed application directly here: **[Ea+UNet Streamlit App](https://edge-attention-resnet-unet.streamlit.app/)**

To run the Streamlit deployment application locally for real-time inference:

```bash
cd "2. Ea+UNet/Streamlit_App"
streamlit run app.py
```

## 📊 Results

The Ea+UNet demonstrates clinical superiority over the baseline UNet, particularly in detecting fuzzy glioblastoma boundaries and reducing false negatives, ensuring higher diagnostic reliability. Detailed comparative analyses can be found in the `Comparison Result` folder.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
