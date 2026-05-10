import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import numpy as np
import matplotlib.pyplot as plt

# Paths
BASE_DIR = r"d:\Major Project"
SIMPLE_UNET_DIR = os.path.join(BASE_DIR, "1. Simple_UNet")
EA_UNET_DIR = os.path.join(BASE_DIR, "2. Ea+UNet")

sys.path.insert(0, SIMPLE_UNET_DIR)
from dataset_brats import BraTSSliceDataset as SimpleDataset
from networks.resnet_decoder import ResNetUNet as SimpleResNetUNet
sys.path.pop(0)

sys.path.insert(0, EA_UNET_DIR)
from data.dataset import BraTSSliceDataset as EaDataset
from models import ResNetUNet as EaResNetUNet
sys.path.pop(0)

def dice_score(pred, target, num_classes=4):
    dices = []
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        intersection = (p & t).sum()
        union = p.sum() + t.sum()
        if union == 0:
            dices.append(1.0)
        else:
            dices.append((2. * intersection) / union)
    return np.mean(dices)

def get_latest_checkpoint(dir_path, size=192):
    import glob
    pattern = os.path.join(dir_path, f"*_{size}", "best_model.pth")
    cands = glob.glob(pattern)
    if not cands:
        pattern = os.path.join(dir_path, f"*_{size}", "last_model.pth")
        cands = glob.glob(pattern)
    if not cands:
        pattern = os.path.join(dir_path, "*", "best_model.pth")
        cands = glob.glob(pattern)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

simple_ckpt = get_latest_checkpoint(os.path.join(SIMPLE_UNET_DIR, "outputs", "Modified_Unet_AISS_brats192"))
ea_ckpt = get_latest_checkpoint(os.path.join(EA_UNET_DIR, "outputs", "EA_ResNetUNet_v2_brats"))

simple_model = SimpleResNetUNet(num_classes=4, input_channels=4).to(device)
if simple_ckpt:
    print(f"Loading Simple_UNet checkpoint: {simple_ckpt}")
    simple_model.load_state_dict(torch.load(simple_ckpt, map_location=device))
else:
    print("Warning: No Simple_UNet checkpoint found!")
simple_model.eval()

ea_model = EaResNetUNet(num_classes=4, input_channels=4).to(device)
if ea_ckpt:
    print(f"Loading Ea+UNet checkpoint: {ea_ckpt}")
    ea_model.load_state_dict(torch.load(ea_ckpt, map_location=device))
else:
    print("Warning: No Ea+UNet checkpoint found!")
ea_model.eval()

# Use EaDataset since it seems identical or compatible, but dataset path can just be the base processed folder
data_path = os.path.join(BASE_DIR, "processed_brats_2d")
if not os.path.exists(data_path):
    # fallback
    data_path = os.path.join(EA_UNET_DIR, "processed_brats_2d")

dataset = EaDataset(
    root_path=data_path,
    split="val",
    img_size=192
)

samples_to_show = 5
# To ensure reproducibility for comparison, let's fix the seed
np.random.seed(42)
# Or pick 5 random slices from the dataset that have some tumor
indices = []
for i in range(min(500, len(dataset))):
    _, label, _ = dataset[i]
    if (label > 0).sum() > 200: # Find slices with some decent tumor area
        indices.append(i)
        if len(indices) >= samples_to_show:
            break

if not indices: # fallback
    indices = np.random.choice(len(dataset), samples_to_show, replace=False)

fig, axes = plt.subplots(samples_to_show, 4, figsize=(16, 4 * samples_to_show))
fig.suptitle("Comparison: Simple_UNet vs Edge-Aware UNet", fontsize=20, y=0.98)
titles = ["FLAIR Image", "Ground Truth", "Simple_UNet Pred", "Ea+UNet Pred"]

# Custom colormap similar to original training scripts if needed, but nipy_spectral works nicely for 4 classes
cmap = plt.get_cmap('nipy_spectral', 4)

for col, title in enumerate(titles):
    axes[0, col].set_title(title, fontsize=14, pad=10)

with torch.no_grad():
    for row, idx in enumerate(indices):
        img, label, meta = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        
        # Simple Unet
        simple_logits = simple_model(img_batch)
        simple_pred = torch.argmax(torch.softmax(simple_logits, dim=1), dim=1).cpu().numpy()[0]
        
        # EA Unet
        ea_logits, _ = ea_model(img_batch)
        ea_pred = torch.argmax(torch.softmax(ea_logits, dim=1), dim=1).cpu().numpy()[0]
        
        img_np = img.numpy()
        flair = img_np[0] # FLAIR is channel 0
        label_np = label.numpy()
        
        simple_dice = dice_score(simple_pred, label_np)
        ea_dice = dice_score(ea_pred, label_np)
        
        axes[row, 0].imshow(flair, cmap='gray')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(label_np, cmap=cmap, vmin=0, vmax=3)
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(simple_pred, cmap=cmap, vmin=0, vmax=3)
        axes[row, 2].text(5, 15, f"Avg Dice: {simple_dice:.3f}", color='white', fontsize=12, backgroundcolor='black')
        axes[row, 2].axis('off')
        
        axes[row, 3].imshow(ea_pred, cmap=cmap, vmin=0, vmax=3)
        axes[row, 3].text(5, 15, f"Avg Dice: {ea_dice:.3f}", color='white', fontsize=12, backgroundcolor='black')
        axes[row, 3].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.94)
out_path = os.path.join(BASE_DIR, "model_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Comparison saved to {out_path}")
