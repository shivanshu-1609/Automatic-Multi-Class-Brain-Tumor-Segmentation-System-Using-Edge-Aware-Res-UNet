import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
base_dir = r"d:\Major Project"
simple_log = os.path.join(base_dir, r"1. Simple_UNet\outputs\Modified_Unet_AISS_brats192\ResNetUNet_epo100_bs48_256\log.csv")
ea_log = os.path.join(base_dir, r"2. Ea+UNet\outputs\EA_ResNetUNet_v2_brats\EA_v2_epo50_bs8_cases500_k10_192\log.csv")

# Load logs
df_simple = pd.read_csv(simple_log)
df_ea = pd.read_csv(ea_log)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Training Curves Comparison: Simple_UNet vs Ea+UNet', fontsize=16)

# Plot Loss
ax1.plot(df_simple['epoch'], df_simple['val_loss'], label='Simple_UNet Val Loss', color='blue', linestyle='-')
ax1.plot(df_ea['epoch'], df_ea['val_loss'], label='Ea+UNet Val Loss', color='red', linestyle='-')

# If training loss exists
ax1.plot(df_simple['epoch'], df_simple['loss'], label='Simple_UNet Train Loss', color='blue', linestyle=':')
ax1.plot(df_ea['epoch'], df_ea['loss'], label='Ea+UNet Train Loss', color='red', linestyle=':')

ax1.set_title('Training & Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot Dice
ax1 = ax2 # repurposing ax2 for Dice
ax1.plot(df_simple['epoch'], df_simple['dice'], label='Simple_UNet Val Dice', color='blue', linestyle='-')
ax1.plot(df_ea['epoch'], df_ea['dice'], label='Ea+UNet Val Dice', color='red', linestyle='-')

ax1.plot(df_simple['epoch'], df_simple['train_dice'], label='Simple_UNet Train Dice', color='blue', linestyle=':')
ax1.plot(df_ea['epoch'], df_ea['train_dice'], label='Ea+UNet Train Dice', color='red', linestyle=':')

ax1.set_title('Training & Validation Dice')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Dice Score')
ax1.legend()
ax1.grid(True)

out_path = os.path.join(base_dir, "training_curves_comparison.png")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
