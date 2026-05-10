import logging
import os
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import BraTSSliceDataset
from utils.losses import BoundaryLoss, GeneralizedDiceLoss
from utils.metrics import compute_mean_dice
from utils.training import (
    AverageMeter, build_epoch_train_loader, metrics_from_meter_state,
    save_training_state, load_training_state, setup_logger, seed_worker,
    initialize_log_csv, append_log_row, read_last_logged_epoch
)
from utils.optimizers import get_optimizer, get_scheduler


def train_one_epoch(
    model, loader, optimizer, ce_loss, dice_loss, edge_loss_fn,
    scaler, device, num_classes, use_amp, use_channels_last,
    epoch, max_epochs, epoch_total_steps,
    checkpoint_callback=None, resume_step_in_epoch=0, resume_meter_state=None,
):
    meters = {"loss": AverageMeter(), "dice": AverageMeter()}

    if resume_meter_state:
        for name, state in resume_meter_state.items():
            if name in meters:
                meters[name].load_state_dict(state)

    model.train()
    progress = tqdm(
        enumerate(loader, start=resume_step_in_epoch),
        total=epoch_total_steps,
        ncols=90,
        initial=resume_step_in_epoch,
        desc=f"Train E{epoch + 1:03d}/{max_epochs:03d}",
    )

    for batch_index, (image_batch, label_batch, _) in progress:
        if use_channels_last and device.type == "cuda":
            image_batch = image_batch.contiguous(memory_format=torch.channels_last)

        image_batch = image_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits, lateral_edge = model(image_batch)

            loss_ce   = ce_loss(logits, label_batch)
            loss_dice = dice_loss(logits, label_batch)

            loss_edge   = edge_loss_fn(lateral_edge, label_batch)

            loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.5 * loss_edge

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dice = compute_mean_dice(logits.detach(), label_batch, num_classes)
        meters["loss"].update(loss.item(), image_batch.size(0))
        meters["dice"].update(dice, image_batch.size(0))

        progress.set_postfix(OrderedDict(
            loss=f"{meters['loss'].avg:.4f}",
            dice=f"{meters['dice'].avg:.4f}",
        ))

        if checkpoint_callback is not None:
            checkpoint_callback(
                epoch=epoch,
                step_in_epoch=batch_index + 1,
                meters=meters,
                phase="train",
            )

    progress.close()
    return {"loss": meters["loss"].avg, "dice": meters["dice"].avg}


def validate(model, loader, ce_loss, dice_loss, edge_loss_fn,
             device, num_classes, use_amp, use_channels_last, epoch, max_epochs):
    meters = {"loss": AverageMeter(), "dice": AverageMeter()}
    edge_meter = AverageMeter()   
    model.eval()

    progress = tqdm(loader, total=len(loader), ncols=90,
                    desc=f"Val   E{epoch + 1:03d}/{max_epochs:03d}")

    with torch.no_grad():
        for image_batch, label_batch, _ in progress:
            if use_channels_last and device.type == "cuda":
                image_batch = image_batch.contiguous(memory_format=torch.channels_last)

            image_batch = image_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, lateral_edge = model(image_batch)

                loss_ce   = ce_loss(logits, label_batch)
                loss_dice = dice_loss(logits, label_batch)

                loss_edge   = edge_loss_fn(lateral_edge, label_batch)

                loss = 0.4 * loss_ce + 0.6 * loss_dice + 0.5 * loss_edge

            dice = compute_mean_dice(logits, label_batch, num_classes)
            meters["loss"].update(loss.item(), image_batch.size(0))
            meters["dice"].update(dice, image_batch.size(0))
            edge_meter.update(loss_edge.item(), image_batch.size(0))

            progress.set_postfix(OrderedDict(
                loss=f"{meters['loss'].avg:.4f}",
                dice=f"{meters['dice'].avg:.4f}",
            ))

    progress.close()
    return {
        "loss":      meters["loss"].avg,
        "dice":      meters["dice"].avg,
        "edge_loss": edge_meter.avg,
    }


def trainer_brats(args, model, snapshot_path):
    os.makedirs(snapshot_path, exist_ok=True)
    setup_logger(snapshot_path)

    logging.info(str(args))
    logging.info("Preparing dataset from root_path=%s", args.root_path)

    device          = args.device
    use_amp         = bool(args.use_amp and device.type == "cuda")
    use_channels_last = bool(getattr(args, "channels_last", False) and device.type == "cuda")
    pin_memory      = device.type == "cuda"

    max_cases = getattr(args, "max_cases", None)  
    atiss_k   = getattr(args, "atiss_k", 10)      

    train_dataset = BraTSSliceDataset(
        root_path    = args.root_path,
        split        = "train",
        img_size     = args.img_size,
        val_split    = args.val_split,
        random_state = 41,
        max_cases    = max_cases,
        atiss_k      = atiss_k,
        augment      = True,
    )
    val_dataset = BraTSSliceDataset(
        root_path    = args.root_path,
        split        = "val",
        img_size     = args.img_size,
        val_split    = args.val_split,
        random_state = 41,
        max_cases    = max_cases,
        atiss_k      = atiss_k,
        augment      = False,
    )

    logging.info("Train slices: %d | Val slices: %d | max_cases=%s | atiss_k=%d",
                 len(train_dataset), len(val_dataset), max_cases, atiss_k)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=partial(seed_worker, base_seed=args.seed),
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"]    = 2

    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    if args.n_gpu > 1 and torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
    model.to(device)

    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Class weights: Background (0) low, NCR/NET (1) high, ED (2) normal, ET (3) high
    class_weights = torch.tensor([0.1, 2.0, 1.0, 1.5], dtype=torch.float32).to(device)
    ce_loss      = nn.CrossEntropyLoss(weight=class_weights).to(device)
    dice_loss    = GeneralizedDiceLoss(args.num_classes).to(device)
    edge_loss_fn = BoundaryLoss().to(device)

    optimizer = get_optimizer(model, base_lr=args.base_lr)
    scheduler = get_scheduler(optimizer, max_epochs=args.max_epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    csv_path   = os.path.join(snapshot_path, "log.csv")
    state_path = os.path.join(snapshot_path, "training_state.pth")
    best_path  = os.path.join(snapshot_path, "best_model.pth")
    last_path  = os.path.join(snapshot_path, "last_model.pth")
    initialize_log_csv(csv_path)

    best_val_dice  = -1.0
    best_val_loss  = float("inf")
    start_epoch    = 0
    resume_step_in_epoch   = 0
    resume_meter_state     = None
    resume_phase           = "train"
    resume_train_metrics   = None

    if args.resume and os.path.exists(state_path):
        resume_state = load_training_state(
            state_path, model, optimizer, scheduler, scaler, device
        )
        best_val_dice       = float(resume_state.get("best_val_dice", best_val_dice))
        best_val_loss       = float(resume_state.get("best_val_loss", best_val_loss))
        phase               = resume_state.get("phase", "train")
        resume_phase        = phase
        resume_train_metrics = resume_state.get("train_metrics")
        last_logged_epoch   = read_last_logged_epoch(csv_path)

        if phase == "train":
            start_epoch          = int(resume_state.get("epoch", 0))
            resume_step_in_epoch = int(resume_state.get("step_in_epoch", 0))
            resume_meter_state   = resume_state.get("meters", {})
        elif phase == "val" and last_logged_epoch >= int(resume_state.get("epoch", 0)) + 1:
            start_epoch, resume_step_in_epoch = int(resume_state.get("epoch", 0)) + 1, 0
            resume_meter_state, resume_phase, resume_train_metrics = None, "train", None
        else:
            start_epoch          = int(resume_state.get("epoch", 0))
            resume_step_in_epoch = 0
            resume_meter_state   = None

        logging.info("Resuming from epoch=%d step=%d phase=%s", start_epoch + 1, resume_step_in_epoch, phase)

    if start_epoch >= args.max_epochs:
        logging.info("Training already reached max_epochs=%d.", args.max_epochs)
        return {"best_val_dice": best_val_dice, "best_val_loss": best_val_loss, "log_csv": csv_path}

    def checkpoint_callback(epoch, step_in_epoch, meters, phase):
        if args.save_every_steps <= 0:
            return
        if phase == "train" and step_in_epoch % args.save_every_steps == 0:
            save_training_state(
                state_path, model, optimizer, scheduler, scaler,
                epoch=epoch, step_in_epoch=step_in_epoch,
                best_val_dice=best_val_dice, best_val_loss=best_val_loss,
                meters=meters, phase=phase,
            )

    for epoch in range(start_epoch, args.max_epochs):
        epoch_resume_step       = resume_step_in_epoch if (epoch == start_epoch and resume_phase == "train") else 0
        epoch_resume_meter_state = resume_meter_state    if (epoch == start_epoch and resume_phase == "train") else None

        train_loader, epoch_total_steps = build_epoch_train_loader(
            train_dataset=train_dataset,
            loader_kwargs=loader_kwargs,
            seed=args.seed,
            epoch=epoch,
            batch_size=args.batch_size,
            resume_step_in_epoch=epoch_resume_step,
        )

        if epoch == start_epoch and resume_phase == "val" and resume_train_metrics is not None:
            train_metrics = resume_train_metrics
            logging.info("Skipping train loop for epoch=%d (at validation boundary).", epoch + 1)
        elif epoch_resume_step >= epoch_total_steps:
            train_metrics = metrics_from_meter_state(epoch_resume_meter_state)
            logging.info("Skipping train loop for epoch=%d (all %d steps done).", epoch + 1, epoch_total_steps)
        else:
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                ce_loss=ce_loss,
                dice_loss=dice_loss,
                edge_loss_fn=edge_loss_fn,
                scaler=scaler,
                device=device,
                num_classes=args.num_classes,
                use_amp=use_amp,
                use_channels_last=use_channels_last,
                epoch=epoch,
                max_epochs=args.max_epochs,
                epoch_total_steps=epoch_total_steps,
                checkpoint_callback=checkpoint_callback,
                resume_step_in_epoch=epoch_resume_step,
                resume_meter_state=epoch_resume_meter_state,
            )

        save_training_state(
            state_path, model, optimizer, scheduler, scaler,
            epoch=epoch, step_in_epoch=0,
            best_val_dice=best_val_dice, best_val_loss=best_val_loss,
            meters=None, phase="val", train_metrics=train_metrics,
        )

        val_metrics = validate(
            model=model, loader=val_loader,
            ce_loss=ce_loss, dice_loss=dice_loss, edge_loss_fn=edge_loss_fn,
            device=device, num_classes=args.num_classes,
            use_amp=use_amp, use_channels_last=use_channels_last,
            epoch=epoch, max_epochs=args.max_epochs,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        log_row = {
            "epoch":      epoch + 1,
            "lr":         current_lr,
            "loss":       train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "val_loss":   val_metrics["loss"],
            "dice":       val_metrics["dice"],
            "edge_loss":  val_metrics.get("edge_loss", 0.0),
        }
        append_log_row(csv_path, log_row)

        logging.info(
            "Epoch %03d | lr %.6f | loss %.4f | train_dice %.4f | val_loss %.4f | val_dice %.4f | edge_loss %.4f",
            epoch + 1, current_lr,
            train_metrics["loss"], train_metrics["dice"],
            val_metrics["loss"],   val_metrics["dice"],
            val_metrics.get("edge_loss", 0.0),
        )

        is_better = (val_metrics["dice"] > best_val_dice) or (
            abs(val_metrics["dice"] - best_val_dice) < 1e-8
            and val_metrics["loss"] < best_val_loss
        )
        state_dict = (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
        torch.save(state_dict, last_path)

        if is_better:
            best_val_dice = val_metrics["dice"]
            best_val_loss = val_metrics["loss"]
            torch.save(state_dict, best_path)
            logging.info("Saved best model → %s", best_path)

        save_training_state(
            state_path, model, optimizer, scheduler, scaler,
            epoch=epoch + 1, step_in_epoch=0,
            best_val_dice=best_val_dice, best_val_loss=best_val_loss,
            meters=None, phase="epoch_boundary",
        )

        resume_step_in_epoch  = 0
        resume_meter_state    = None
        resume_phase          = "train"
        resume_train_metrics  = None

    return {"best_val_dice": best_val_dice, "best_val_loss": best_val_loss, "log_csv": csv_path}
