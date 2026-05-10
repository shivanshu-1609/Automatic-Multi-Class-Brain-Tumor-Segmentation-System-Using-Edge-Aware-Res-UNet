import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from networks.resnet_decoder import ResNetUNet
from performance_profile import auto_num_workers, auto_train_batch_size
from trainer_brats import trainer_brats
from visualize import plot_training_curve


# Set up base directories for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Default paths for raw and preprocessed data
DEFAULT_RAW_ROOT_PATH = os.path.join(REPO_ROOT, "BraTS2021_Training_Data")
DEFAULT_PROCESSED_ROOT_PATH = os.path.join(BASE_DIR, "processed_brats_2d")

def resolve_default_root_path():
    """Select preprocessed 2D data if available to speed up data loading"""
    if os.path.isdir(DEFAULT_PROCESSED_ROOT_PATH):
        return DEFAULT_PROCESSED_ROOT_PATH
    return DEFAULT_RAW_ROOT_PATH

DEFAULT_ROOT_PATH = resolve_default_root_path()

def str2bool(value):
    """Helper to parse boolean command line arguments"""
    if isinstance(value, bool): return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}: return True
    if value in {"false", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

# Training Configuration Parameters
parser = argparse.ArgumentParser(description="BraTS Model Training Script")
parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT_PATH, help="Path to the dataset")
parser.add_argument("--dataset", type=str, default="brats", help="Dataset identifier")
parser.add_argument("--num_classes", type=int, default=4, help="Number of segmentation classes (including background)")
parser.add_argument("--input_channels", type=int, default=4, help="Number of MRI modalities (FLAIR, T1, T1ce, T2)")
parser.add_argument("--max_epochs", type=int, default=100, help="Total number of training epochs")
parser.add_argument("--batch_size", type=int, default=0, help="Training batch size (0 for auto-tuning)")
parser.add_argument("--base_lr", type=float, default=0.01, help="Initial learning rate")
parser.add_argument("--img_size", type=int, default=224, help="Input spatial resolution (squared)")
parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data used for validation")
parser.add_argument("--use_amp", type=str2bool, default=True, help="Enable Automatic Mixed Precision for faster training")
parser.add_argument("--num_workers", type=int, default=-1, help="Number of data loader sub-processes (-1 for auto)")
parser.add_argument("--channels_last", type=str2bool, default=True, help="Use memory-efficient channels_last format on CUDA")
parser.add_argument("--use_tf32", type=str2bool, default=True, help="Enable TF32 on Ampere GPUs for acceleration")
parser.add_argument("--name", type=str, default="Modified_Unet_ATISS_Run", help="Experiment name for output folder")
parser.add_argument("--resume", type=str2bool, default=True, help="Resume training if a checkpoint exists")

args = parser.parse_args()

if __name__ == "__main__":
    start_time = time.time()
    print(f"Initializing training at {datetime.now()}")
    print(f"PyTorch version: {torch.__version__} | CUDA: {torch.version.cuda}")

    # Set up hardware acceleration settings
    if torch.cuda.is_available():
        cudnn.benchmark = True # Optimized for fixed input sizes
        if args.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
    
    # Ensure reproducibility across runs
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-tuning resources if not specified
    if args.num_workers < 0: args.num_workers = auto_num_workers()
    if args.batch_size <= 0: args.batch_size = auto_train_batch_size(args.img_size)

    print(f"Device: {args.device} | Batch Size: {args.batch_size} | Workers: {args.num_workers}")

    # Prepare output directory for this experiment
    exp_folder = f"{args.name}_bs{args.batch_size}_{args.img_size}"
    snapshot_path = os.path.join(BASE_DIR, "outputs", exp_folder)
    os.makedirs(snapshot_path, exist_ok=True)

    # Initialize the Modified Unet Model
    model = ResNetUNet(
        num_classes=args.num_classes,
        input_channels=args.input_channels
    ).to(args.device)

    # Convert model to channels_last for better inference/training speed on modern GPUs
    if args.channels_last and args.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    # Execute the training pipeline
    training_summary = trainer_brats(args, model, snapshot_path)

    # Generate training visualization (Loss and Dice curves)
    plot_training_curve(
        training_summary["log_csv"],
        os.path.join(snapshot_path, "training_curve.png")
    )

    total_time = (time.time() - start_time) / 60
    print(f"Training completed in {total_time:.2f} minutes.")
    print(f"Best Validation Dice Score: {training_summary['best_val_dice']:.4f}")
