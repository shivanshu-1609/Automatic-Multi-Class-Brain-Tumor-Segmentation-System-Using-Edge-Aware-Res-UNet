import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import torch
import numpy as np
import cv2
import sys
from PIL import Image
import matplotlib.pyplot as plt

# Add current directory to path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models import ResNetUNet

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & UI SETUP
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BraTS Edge-Aware Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean medical UI
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #4CAF50; color: white; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    h1 { color: #1e3d59; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stAlert { border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🧠 BraTS Medical Image Segmentation")
st.markdown("### Edge-Attention ResNet-UNet v2 Deployment")
st.info("Upload 4 MRI modalities (T1, T1ce, T2, FLAIR) to generate tumor segmentation and boundary maps.")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(num_classes=4, input_channels=4).to(device)
    checkpoint_path = os.path.join(BASE_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.error(f"Checkpoint not found at {checkpoint_path}. Please check the folder.")
        return None, None

model, device = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(modalities_list):
    """
    Expects a list of 4 PIL images or numpy arrays.
    Returns a torch tensor of shape [1, 4, 192, 192]
    """
    processed = []
    for img in modalities_list:
        # Resize to 192x192
        img_res = cv2.resize(np.array(img), (192, 192))
        # Normalize 0-1
        img_norm = (img_res - img_res.min()) / (img_res.max() - img_res.min() + 1e-8)
        processed.append(img_norm)
    
    # Stack to [4, 192, 192]
    stacked = np.stack(processed, axis=0).astype(np.float32)
    return torch.from_numpy(stacked).unsqueeze(0).to(device)

def get_overlay(base_img, mask, alpha=0.5):
    """Create a colored overlay of the mask on the base image."""
    # Class colors: [Background=None, NCR/NET=Red, ED=Green, ET=Yellow]
    colors = [
        [0, 0, 0],       # Background
        [255, 0, 0],     # NCR/NET (Red)
        [0, 255, 0],     # ED (Green)
        [255, 255, 0]    # ET (Yellow)
    ]
    
    # Base image to 3-channel
    if len(base_img.shape) == 2:
        base_rgb = cv2.cvtColor((base_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = base_img.copy()

    overlay = base_rgb.copy()
    for c in range(1, 4):
        overlay[mask == c] = colors[c]
    
    return cv2.addWeighted(base_rgb, 1 - alpha, overlay, alpha, 0)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR: UPLOAD & SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.header("📁 Data Input")
upload_t1    = st.sidebar.file_uploader("T1 Modality", type=['png', 'jpg', 'jpeg', 'npy'])
upload_t1ce  = st.sidebar.file_uploader("T1ce Modality", type=['png', 'jpg', 'jpeg', 'npy'])
upload_t2    = st.sidebar.file_uploader("T2 Modality", type=['png', 'jpg', 'jpeg', 'npy'])
upload_flair = st.sidebar.file_uploader("FLAIR Modality", type=['png', 'jpg', 'jpeg', 'npy'])

overlay_alpha = st.sidebar.slider("Mask Opacity", 0.0, 1.0, 0.4)

st.sidebar.markdown("---")
st.sidebar.markdown("### Class Legend")
st.sidebar.markdown("🔴 **NCR/NET**: Necrotic/Non-Enhancing Core")
st.sidebar.markdown("🟢 **ED**: Peritumoral Edema")
st.sidebar.markdown("🟡 **ET**: Enhancing Tumor")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if upload_t1 and upload_t1ce and upload_t2 and upload_flair:
    # Load images
    try:
        imgs = []
        for up in [upload_t1, upload_t1ce, upload_t2, upload_flair]:
            if up.name.endswith('.npy'):
                imgs.append(np.load(up))
            else:
                imgs.append(Image.open(up).convert('L'))
        
        # Preprocess
        input_tensor = preprocess_image(imgs)
        
        with st.spinner("🧠 Segmenting..."):
            with torch.no_grad():
                logits, lateral_edge = model(input_tensor)
            
            # Predictions
            pred_mask = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
            edge_prob = torch.sigmoid(lateral_edge)[0, 0].cpu().numpy()
        
        # Visuals
        flair_norm = (np.array(imgs[3]) - np.array(imgs[3]).min()) / (np.array(imgs[3]).max() - np.array(imgs[3]).min() + 1e-8)
        flair_res  = cv2.resize(flair_norm, (192, 192))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image(flair_res, caption="Input FLAIR", use_container_width=True)
            
        with col2:
            overlay_img = get_overlay(flair_res, pred_mask, alpha=overlay_alpha)
            st.image(overlay_img, caption="Segmentation Prediction", use_container_width=True)
            
        with col3:
            # Colorize edge map
            edge_vis = plt.cm.hot(edge_prob)[:, :, :3]
            st.image(edge_vis, caption="Edge Attention Map", use_container_width=True)
            
        st.success("Analysis Complete!")
        
        # Detailed Stats
        st.markdown("---")
        st.markdown("### 📊 Region Analysis")
        c1, c2, c3 = st.columns(3)
        total_px = pred_mask.size
        c1.metric("NCR/NET Pixels", np.sum(pred_mask == 1))
        c2.metric("Edema Pixels", np.sum(pred_mask == 2))
        c3.metric("Enhancing Tumor Pixels", np.sum(pred_mask == 3))

    except Exception as e:
        st.error(f"Error processing files: {e}")

else:
    st.warning("Please upload all 4 MRI modalities in the sidebar to start.")
    
    # Placeholder for UI look
    st.image("https://via.placeholder.com/1200x400.png?text=Waiting+for+Input+Files...", use_container_width=True)

st.markdown("---")
st.caption("Developed for BraTS 2021 Segmentation | Edge-Aware ResNet-UNet v2")
