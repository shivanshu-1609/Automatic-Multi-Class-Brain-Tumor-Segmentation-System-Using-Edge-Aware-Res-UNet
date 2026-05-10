import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_brats import BraTSSliceDataset
from networks.resnet_decoder import ResNetUNet
from performance_profile import auto_inference_batch_size, auto_num_workers
from visualize import plot_predictions, save_class_masks, save_segmentation_mask


# Directory for current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)

# Default dataset paths
DEFAULT_RAW_ROOT_PATH = os.path.join(REPO_ROOT, "BraTS2021_Training_Data")
DEFAULT_PROCESSED_ROOT_PATH = os.path.join(BASE_DIR, "processed_brats_2d")

def resolve_default_root_path():
    """Prefer processed 2D dataset if available for direct slice-based inference"""
    if os.path.isdir(DEFAULT_PROCESSED_ROOT_PATH):
        return DEFAULT_PROCESSED_ROOT_PATH
    return DEFAULT_RAW_ROOT_PATH

DEFAULT_ROOT_PATH = resolve_default_root_path()
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "inference_outputs")

def str2bool(value):
    if isinstance(value, bool): return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}: return True
    if value in {"false", "0", "no", "n"}: return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

# Argument parser for testing settings
parser = argparse.ArgumentParser(description="BraTS Inference Script")
parser.add_argument("--root_path", type=str, default=DEFAULT_ROOT_PATH, help="Dataset root path")
parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model .pth")
parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Dataset split to test")
parser.add_argument("--num_classes", type=int, default=4, help="Number of output classes")
parser.add_argument("--input_channels", type=int, default=4, help="Number of MRI modalities")
parser.add_argument("--img_size", type=int, default=224, help="Input image size")
parser.add_argument("--batch_size", type=int, default=0, help="Batch size (0 for auto)")
parser.add_argument("--num_workers", type=int, default=-1, help="Dataloader workers (-1 for auto)")
parser.add_argument("--seed", type=int, default=1234, help="Random seed")
parser.add_argument("--save_visuals", type=str2bool, default=True, help="Save overlay and class mask PNGs")
parser.add_argument("--save_eval_pairs", type=str2bool, default=True, help="Save .npy pairs for independent evaluation script")
parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Inference output directory")

args = parser.parse_args()

def resolve_auto_checkpoint(img_size):
    """Automatically find the best model checkpoint based on img_size"""
    exp_dir = os.path.join(BASE_DIR, "outputs", "Modified_Unet_AISS_brats192")
    pattern = os.path.join(exp_dir, f"ResNetUNet_epo*_bs*_{img_size}", "best_model.pth")
    candidates = glob.glob(pattern)
    if not candidates: return ""
    return max(candidates, key=os.path.getmtime)

def run_test(args, model, loader, device):
    """Run inference, save prediction results for independent evaluation"""
    model.eval()
    
    # Setup directories for visual output
    vis_pred_dir = os.path.join(args.output_dir, "predictions")
    vis_overlay_dir = os.path.join(args.output_dir, "overlays")
    vis_class_dir = os.path.join(args.output_dir, "class_masks")
    
    # Setup directories for standard evaluation tool
    eval_gt_dir = os.path.join(BASE_DIR, "evaluation_metrics", "GT")
    eval_pred_dir = os.path.join(BASE_DIR, "evaluation_metrics", "pred")

    for d in [vis_pred_dir, vis_overlay_dir, vis_class_dir, eval_gt_dir, eval_pred_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"Starting inference on {len(loader.dataset)} samples...")
    
    with torch.no_grad():
        for image_batch, label_batch, meta in tqdm(loader, desc="Testing"):
            image_batch = image_batch.to(device, non_blocking=True)
            
            # Predict
            logits = model(image_batch)
            pred_batch = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
            
            image_np = image_batch.cpu().numpy()
            label_np = label_batch.numpy()

            for i in range(pred_batch.shape[0]):
                p_mask = pred_batch[i].astype(np.uint8)
                l_mask = label_np[i].astype(np.uint8)
                img = image_np[i]
                
                case_id = meta["case_id"][i]
                slice_idx = int(meta["slice_index"][i])
                sample_name = f"{case_id}_slice_{slice_idx:03d}"

                # 1. Save data for Evaluation Script (optional, for independent metrics)
                if args.save_eval_pairs:
                    np.save(os.path.join(eval_gt_dir, f"{sample_name}.npy"), l_mask)
                    np.save(os.path.join(eval_pred_dir, f"{sample_name}.npy"), p_mask)

                # 2. Save Visual Comparisons if requested
                if args.save_visuals:
                    save_segmentation_mask(p_mask, os.path.join(vis_pred_dir, f"{sample_name}_pred.png"))
                    plot_predictions(img, l_mask, p_mask, os.path.join(vis_overlay_dir, f"{sample_name}_overlay.png"))
                    save_class_masks(p_mask, os.path.join(vis_class_dir, sample_name), prefix=sample_name)

    print(f"Inference complete. Visuals saved to {args.output_dir}.")
    if args.save_eval_pairs:
        print(f"Evaluation pairs (.npy) saved to {eval_pred_dir} for independent evaluation.")

if __name__ == "__main__":
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-resolving params
    if args.num_workers < 0: args.num_workers = auto_num_workers()
    if args.batch_size <= 0: args.batch_size = auto_inference_batch_size(args.img_size)
    if not args.checkpoint: args.checkpoint = resolve_auto_checkpoint(args.img_size)

    if not args.checkpoint or not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"No checkpoint found at {args.checkpoint}. Please train first.")

    # Load Model
    model = ResNetUNet(num_classes=args.num_classes, input_channels=args.input_channels).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Load Dataset
    dataset = BraTSSliceDataset(root_path=args.root_path, split=args.split, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    run_test(args, model, loader, device)
