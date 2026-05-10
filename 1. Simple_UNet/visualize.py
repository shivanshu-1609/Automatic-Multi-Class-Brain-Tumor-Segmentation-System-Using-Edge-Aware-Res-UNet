import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Class labels for BraTS segmentation
BRATS_CLASS_NAMES = ["Background", "NCR/NET", "ED", "ET"]

# RGB color mapping for each class
BRATS_COLOR_MAP = np.array(
    [
        [0, 0, 0],        # Background
        [220, 20, 60],    # NCR/NET
        [30, 144, 255],   # ED
        [50, 205, 50],    # ET
    ],
    dtype=np.uint8,
)


def mask_to_color(mask):
    # Convert class indices to RGB colors using predefined map
    clipped_mask = np.clip(mask.astype(np.int64), 0, len(BRATS_COLOR_MAP) - 1)
    return BRATS_COLOR_MAP[clipped_mask]


def _prepare_background_image(image):
    # Normalize image slice for better visualization (0–1 range)
    if image.ndim == 3:
        base = image[-1]  # take last channel (FLAIR usually)
    else:
        base = image

    base = base.astype(np.float32)
    base_min = base.min()
    base_max = base.max()

    if base_max - base_min < 1e-8:
        return np.zeros_like(base)

    return (base - base_min) / (base_max - base_min)


def save_segmentation_mask(mask, output_path):
    # Save full segmentation mask as color image
    plt.imsave(output_path, mask_to_color(mask))


def save_class_masks(pred, output_dir, prefix="pred", class_names=None):
    # Save binary mask for each tumor class separately
    class_names = class_names or BRATS_CLASS_NAMES
    os.makedirs(output_dir, exist_ok=True)

    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue  # skip background

        class_mask = (pred == class_index).astype(np.uint8) * 255

        file_name = f"{prefix}_{class_name.lower().replace('/', '_').replace(' ', '_')}.png"

        plt.imsave(
            os.path.join(output_dir, file_name),
            class_mask,
            cmap="gray",
            vmin=0,
            vmax=255,
        )


def plot_predictions(image, gt, pred, output_path, class_names=None):
    # Create side-by-side visualization: input, ground truth, prediction
    class_names = class_names or BRATS_CLASS_NAMES
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    background = _prepare_background_image(image)
    gt_color = mask_to_color(gt)
    pred_color = mask_to_color(pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original slice
    axes[0].imshow(background, cmap="gray")
    axes[0].set_title("FLAIR Slice")

    # Ground truth overlay
    axes[1].imshow(background, cmap="gray")
    axes[1].imshow(gt_color, alpha=0.45)
    axes[1].set_title("Ground Truth Overlay")

    # Prediction overlay
    axes[2].imshow(background, cmap="gray")
    axes[2].imshow(pred_color, alpha=0.45)
    axes[2].set_title("Prediction Overlay")

    for axis in axes:
        axis.axis("off")

    # Legend for tumor classes
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label=class_name,
            markerfacecolor=color / 255.0,
            markersize=10,
        )
        for class_name, color in zip(class_names[1:], BRATS_COLOR_MAP[1:])
    ]

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=max(1, len(legend_handles)))

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_training_curve(log_csv, output_path=None):
    # Plot training vs validation loss and Dice score from log CSV
    log_data = pd.read_csv(log_csv)

    output_path = output_path or os.path.join(
        os.path.dirname(log_csv),
        "training_curve.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].plot(log_data["epoch"], log_data["loss"], label="Train Loss")
    axes[0].plot(log_data["epoch"], log_data["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")

    # Dice score curves
    dice_column = "dice" if "dice" in log_data.columns else "val_dice"
    axes[1].plot(log_data["epoch"], log_data[dice_column], label="Validation Dice")

    if "train_dice" in log_data.columns:
        axes[1].plot(log_data["epoch"], log_data["train_dice"], label="Train Dice")

    axes[1].set_title("Dice Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path