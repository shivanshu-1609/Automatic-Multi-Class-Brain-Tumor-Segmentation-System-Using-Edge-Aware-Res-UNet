import os
import csv
import logging
import random
import torch
import numpy as np
from torch.utils.data import Sampler, DataLoader

# =========================================================================
# Metric Logging Utilities
# =========================================================================
class AverageMeter:
    """Tracks running mean of a scalar metric across batches (for logging)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = float(val)
        self.sum   += float(val) * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)

    def state_dict(self):
        return {"val": self.val, "avg": self.avg, "sum": self.sum, "count": self.count}

    def load_state_dict(self, state):
        self.val   = float(state.get("val",   0.0))
        self.avg   = float(state.get("avg",   0.0))
        self.sum   = float(state.get("sum",   0.0))
        self.count = int(state.get("count",   0))

def metrics_from_meter_state(meter_state):
    """Reconstruct epoch-average metrics from a saved AverageMeter state dict."""
    meters = {"loss": AverageMeter(), "dice": AverageMeter()}
    for name, state in (meter_state or {}).items():
        if name in meters:
            meters[name].load_state_dict(state)
    return {"loss": meters["loss"].avg, "dice": meters["dice"].avg}

# =========================================================================
# DataLoader & Determinism Utilities
# =========================================================================
class FixedIndexSampler(Sampler):
    """Deterministic sampler for exact mid-epoch checkpointing and resume."""
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def seed_worker(worker_id, base_seed):
    """Seed DataLoader workers for reproducible augmentation / shuffling."""
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def build_epoch_train_loader(train_dataset, loader_kwargs, seed, epoch, batch_size, resume_step_in_epoch=0):
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    all_indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    total_steps   = (len(all_indices) + batch_size - 1) // batch_size
    resume_offset = min(resume_step_in_epoch * batch_size, len(all_indices))
    remaining     = all_indices[resume_offset:]

    train_loader = DataLoader(
        train_dataset,
        shuffle=False,
        sampler=FixedIndexSampler(remaining),
        **loader_kwargs,
    )
    return train_loader, total_steps

# =========================================================================
# Checkpoint Saving & Loading
# =========================================================================
def save_training_state(
    checkpoint_path, model, optimizer, scheduler, scaler,
    epoch, step_in_epoch, best_val_dice, best_val_loss,
    meters=None, phase="train", train_metrics=None,
):
    import torch.nn as nn
    state_dict = (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())

    checkpoint = {
        "epoch":              epoch,
        "step_in_epoch":      step_in_epoch,
        "best_val_dice":      best_val_dice,
        "best_val_loss":      best_val_loss,
        "phase":              phase,
        "model_state_dict":   state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":  scaler.state_dict(),
        "meters":             {n: m.state_dict() for n, m in (meters or {}).items()},
        "train_metrics":      train_metrics,
    }
    torch.save(checkpoint, checkpoint_path)

def load_training_state(checkpoint_path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    return checkpoint

# =========================================================================
# Logger & CSV Creation
# =========================================================================
import sys
def setup_logger(snapshot_path):
    log_file = os.path.join(snapshot_path, "log.txt")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = []

    fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root.addHandler(fh)
    root.addHandler(sh)

_CSV_FIELDS = ["epoch", "lr", "loss", "train_dice", "val_loss", "dice", "edge_loss"]

def initialize_log_csv(csv_path):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

def append_log_row(csv_path, log_row):
    import time
    for attempt in range(60):
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(log_row)
            return
        except PermissionError:
            if attempt == 0:
                logging.warning("log.csv is locked. Please close it! Retrying...")
            time.sleep(5)
    logging.error("Failed to write to log.csv after retries. Skipping.")

def read_last_logged_epoch(csv_path):
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return int(rows[-1]["epoch"]) if rows else 0
    except Exception:
        return 0
