"""Fine-tune SLFNet with validation, AMP, clipping, and early stopping."""

import argparse
import math
import os
import random
import time

import numpy as np
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


cli_args = parse_args()
with open(cli_args.config, encoding="utf-8") as handle:
    args = yaml.safe_load(handle)
os.environ["CUDA_VISIBLE_DEVICES"] = args["train"]["gpus_id"]

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from dataset.sdg_depth import SDGDepth
from metrics import Metrics
from models import get_model
from models.SLFNet.loss import DesignLoss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_weights(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    for key in ("state_dict", "net", "model"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            checkpoint = checkpoint[key]
            break
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model.load_state_dict(state)


def move_to_cuda(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def compute_training_loss(model, loss_function, batch):
    outputs = model(batch)
    (
        pam_disp,
        mask_disp,
        spn_inter,
        pt_disp,
        pam_mask,
        valid_left_to_right,
        valid_right_to_left,
        matrix_left_to_right,
        matrix_right_to_left,
        matrix_left_right_left,
        matrix_right_left_right,
    ) = outputs
    return loss_function(
        (pam_disp, mask_disp, spn_inter, pt_disp),
        pam_mask,
        batch["gt_depth"],
        batch["K"],
        [
            (matrix_right_to_left, matrix_left_to_right),
            (matrix_left_right_left, matrix_right_left_right),
            (valid_left_to_right, valid_right_to_left),
        ],
        (batch["rgb1"], batch["rgb2"]),
    )


def validation_metrics(model, loader, metric):
    model.eval()
    totals = np.zeros(8, dtype=np.float64)
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_to_cuda(batch)
            output = model(batch)
            focal = batch["K"][:, 1, None, None, None]
            baseline = batch["K"][:, 0, None, None, None]
            prediction = torch.clamp(
                focal * baseline / output["disp"].clamp_min(1e-8),
                1,
                args["model"]["max_depth"],
            )
            totals += metric(prediction, batch["gt_depth"])[0].cpu().numpy()
            count += 1
    return totals / max(count, 1)


def learning_rate(epoch, total_epochs, base_lr, warmup_epochs, min_lr):
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


class TrainingLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, message):
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SLFNet fine-tuning")
    seed_everything(args["environment"]["seed"])
    torch.backends.cudnn.deterministic = args["environment"]["reemerge"]
    torch.backends.cudnn.benchmark = not args["environment"]["reemerge"]

    train_set = SDGDepth(args, "train")
    val_set = SDGDepth(args, "val")
    train_loader = DataLoader(
        train_set,
        batch_size=args["train"]["batch_size"],
        shuffle=True,
        num_workers=args["train"]["workers"],
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args["test"]["batch_size"],
        shuffle=False,
        num_workers=args["test"]["workers"],
        pin_memory=True,
    )

    model = get_model(args["model"]["model_name"])(args).cuda()
    load_weights(model, args["path"]["loadmodel"])
    loss_function = DesignLoss(args).cuda()
    metric = Metrics(args).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["train"]["lr"],
        betas=(0.9, 0.999),
        weight_decay=args["train"]["weight_decay"],
    )
    scaler = GradScaler(enabled=args["train"]["amp"])
    save_dir = args["path"]["savemodel"]
    os.makedirs(save_dir, exist_ok=True)
    log_path = args["path"].get("train_log", os.path.join(save_dir, "train.log"))
    logger = TrainingLogger(log_path)
    logger(f"TRAINING_START config={cli_args.config} dry_run={cli_args.dry_run}")

    best_rmse = float("inf")
    stale_epochs = 0
    total_epochs = 1 if cli_args.dry_run else args["train"]["max_epochs"]
    for epoch in range(total_epochs):
        lr = learning_rate(
            epoch,
            args["train"]["max_epochs"],
            args["train"]["lr"],
            args["train"]["warmup_epochs"],
            args["train"]["min_lr"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        epoch_loss = 0.0
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch = move_to_cuda(batch)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=args["train"]["amp"]):
                loss = compute_training_loss(model, loss_function, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch + 1}, step {step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = clip_grad_norm_(
                model.parameters(), args["train"]["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if cli_args.dry_run:
                logger(
                    f"DRY_RUN_OK loss={loss.item():.6f} "
                    f"grad_norm={float(gradient_norm):.6f} "
                    f"peak_memory_gb={torch.cuda.max_memory_allocated()/2**30:.3f}"
                )
                return

        scores = validation_metrics(model, val_loader, metric)
        train_loss = epoch_loss / len(train_loader)
        rmse_mm = float(scores[0] * 1000)
        logger(
            f"epoch={epoch + 1:03d} lr={lr:.7f} train_loss={train_loss:.6f} "
            f"val_rmse_mm={rmse_mm:.3f} val_mae_mm={scores[1]*1000:.3f} "
            f"val_rel={scores[4]:.6f} time_s={time.time()-started:.1f}"
        )
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_metrics": scores,
            "config": args,
        }
        torch.save(checkpoint, os.path.join(save_dir, "latest.tar"))
        if (epoch + 1) % args["train"]["save_every"] == 0:
            torch.save(checkpoint, os.path.join(save_dir, f"epoch_{epoch+1:03d}.tar"))
        if rmse_mm < best_rmse:
            best_rmse = rmse_mm
            stale_epochs = 0
            torch.save(checkpoint, os.path.join(save_dir, "best.tar"))
        else:
            stale_epochs += 1
            if stale_epochs >= args["train"]["early_stopping_patience"]:
                logger(f"early_stop epoch={epoch + 1} best_rmse_mm={best_rmse:.3f}")
                break
    logger(f"TRAINING_DONE best_rmse_mm={best_rmse:.3f}")


if __name__ == "__main__":
    main()
