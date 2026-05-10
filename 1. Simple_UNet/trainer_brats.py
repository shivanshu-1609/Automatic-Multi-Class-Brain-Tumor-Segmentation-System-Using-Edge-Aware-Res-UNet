import csv
import logging
import os
import random
import sys
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from dataset_brats import BraTSSliceDataset

# --- Helper Classes and Functions ---

class AverageMeter:
    """Computes and stores the average and current value of a metric"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0.0; self.avg = 0.0; self.sum = 0.0; self.count = 0
    def update(self, val, n=1):
        self.val = float(val); self.sum += float(val) * n; self.count += n
        self.avg = self.sum / max(self.count, 1)
    def state_dict(self):
        return {"val": self.val, "avg": self.avg, "sum": self.sum, "count": self.count}
    def load_state_dict(self, state):
        self.val = float(state.get("val", 0.0)); self.avg = float(state.get("avg", 0.0))
        self.sum = float(state.get("sum", 0.0)); self.count = int(state.get("count", 0))

class FixedIndexSampler(Sampler):
    """Sampler that yields indices in a fixed order, used for deterministic resuming"""
    def __init__(self, indices):
        self.indices = list(indices)
    def __iter__(self): return iter(self.indices)
    def __len__(self): return len(self.indices)

def seed_worker(worker_id, base_seed):
    """Ensure each dataloader worker has a unique but reproducible seed"""
    seed = base_seed + worker_id
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# --- Loss Functions ---

class DiceLoss(nn.Module):
    """Multi-class Dice Loss implementation"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            tensor_list.append((input_tensor == i).unsqueeze(1))
        return torch.cat(tensor_list, dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax: inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None: weight = [1] * self.n_classes
        
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i]) * weight[i]
        return loss / self.n_classes

def compute_mean_dice(logits, target, num_classes):
    """Compute mean Dice score for foreground classes (excluding background)"""
    pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    dices = []
    for class_index in range(1, num_classes): # Class 0 is background
        pred_mask = (pred == class_index).float()
        target_mask = (target == class_index).float()
        intersection = torch.sum(pred_mask * target_mask)
        denominator = pred_mask.sum() + target_mask.sum()
        if denominator.item() == 0:
            dices.append(1.0) # Correctly predicted empty class
            continue
        dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5)
        dices.append(dice.item())
    return float(np.mean(dices)) if dices else 0.0

# --- Training / Validation Core ---

def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, scaler, device, num_classes, use_amp, epoch, max_epochs, total_steps, resume_step=0):
    """Execute one training epoch"""
    meters = {"loss": AverageMeter(), "dice": AverageMeter()}
    model.train()
    
    progress = tqdm(enumerate(loader, start=resume_step), total=total_steps, ncols=100, initial=resume_step,
                    desc=f"Train E{epoch+1:03d}/{max_epochs:03d}")

    for idx, (images, labels, _) in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = 0.5 * ce_loss(logits, labels) + 0.5 * dice_loss(logits, labels, softmax=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        dice = compute_mean_dice(logits.detach(), labels, num_classes)
        meters["loss"].update(loss.item(), images.size(0))
        meters["dice"].update(dice, images.size(0))
        
        progress.set_postfix(loss=f"{meters['loss'].avg:.4f}", dice=f"{meters['dice'].avg:.4f}")

    return {"loss": meters["loss"].avg, "dice": meters["dice"].avg}

def validate(model, loader, ce_loss, dice_loss, device, num_classes, use_amp, epoch, max_epochs):
    """Run validation on the evaluation set"""
    meters = {"loss": AverageMeter(), "dice": AverageMeter()}
    model.eval()
    
    progress = tqdm(loader, total=len(loader), ncols=100, desc=f"Val   E{epoch+1:03d}/{max_epochs:03d}")

    with torch.no_grad():
        for images, labels, _ in progress:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = 0.5 * ce_loss(logits, labels) + 0.5 * dice_loss(logits, labels, softmax=True)
            
            dice = compute_mean_dice(logits, labels, num_classes)
            meters["loss"].update(loss.item(), images.size(0))
            meters["dice"].update(dice, images.size(0))
            progress.set_postfix(loss=f"{meters['loss'].avg:.4f}", dice=f"{meters['dice'].avg:.4f}")

    return {"loss": meters["loss"].avg, "dice": meters["dice"].avg}

# --- Main Pipeline ---

def trainer_brats(args, model, snapshot_path):
    """Main training coordinator"""
    os.makedirs(snapshot_path, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(snapshot_path, "log.txt")
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', 
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"Start training with config: {args}")

    device = args.device
    use_amp = bool(args.use_amp and device.type == "cuda")

    # Load Data
    train_ds = BraTSSliceDataset(root_path=args.root_path, split="train", img_size=args.img_size, val_split=args.val_split)
    val_ds = BraTSSliceDataset(root_path=args.root_path, split="val", img_size=args.img_size, val_split=args.val_split)
    
    loader_params = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                         worker_init_fn=partial(seed_worker, base_seed=args.seed))
    
    val_loader = DataLoader(val_ds, shuffle=False, **loader_params)

    # Model, Optimizer, Scheduler
    ce_loss = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss(args.num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    csv_path = os.path.join(snapshot_path, "log.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["epoch", "lr", "loss", "train_dice", "val_loss", "dice"]).writeheader()

    best_dice = 0.0
    
    for epoch in range(args.max_epochs):
        # Shuffled loader for every epoch
        gen = torch.Generator(); gen.manual_seed(args.seed + epoch)
        indices = torch.randperm(len(train_ds), generator=gen).tolist()
        train_loader = DataLoader(train_ds, sampler=FixedIndexSampler(indices), **loader_params)
        
        # Train
        train_stats = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, scaler, device, 
                                      args.num_classes, use_amp, epoch, args.max_epochs, len(train_loader))
        
        # Validate
        val_stats = validate(model, val_loader, ce_loss, dice_loss, device, args.num_classes, use_amp, epoch, args.max_epochs)
        
        scheduler.step()
        
        # Logging results
        res = {"epoch": epoch+1, "lr": optimizer.param_groups[0]["lr"], "loss": train_stats["loss"], 
               "train_dice": train_stats["dice"], "val_loss": val_stats["loss"], "dice": val_stats["dice"]}
        
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=res.keys()).writerow(res)
            
        logging.info(f"E{epoch+1:03d} | Loss: {res['loss']:.4f} | Val Dice: {res['dice']:.4f}")

        # Save Checkpoints
        ckpt = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(ckpt, os.path.join(snapshot_path, "last_model.pth"))
        if res["dice"] > best_dice:
            best_dice = res["dice"]
            torch.save(ckpt, os.path.join(snapshot_path, "best_model.pth"))
            logging.info(f"New best model saved with Dice: {best_dice:.4f}")

    return {"best_val_dice": best_dice, "log_csv": csv_path}
