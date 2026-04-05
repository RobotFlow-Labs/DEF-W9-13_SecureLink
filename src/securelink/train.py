"""SecureLink training pipeline.

Training flow:
  1. Load config from TOML
  2. Build datasets and dataloaders
  3. Build model, optimizer, scheduler
  4. Training loop with multi-similarity loss
  5. After DNN training: fit OC-SVM per UAV
  6. Save complete model (DNN + OC-SVMs)
"""

from __future__ import annotations

import math
import pickle
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from securelink.dataset import (
    build_dataloaders,
    load_all_data,
    split_dataset_a,
)
from securelink.losses import MultiSimilarityLoss
from securelink.model import SecureLinkModel, build_model
from securelink.utils import (
    SecureLinkConfig,
    ensure_artifact_dirs,
    get_device,
    seed_everything,
)

# ---------------------------------------------------------------------------
# Learning rate scheduler: warmup + cosine
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Warmup + cosine annealing learning rate scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {"current_step": self.current_step}

    def load_state_dict(self, state: dict) -> None:
        self.current_step = state["current_step"]


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages model checkpoints, keeping only top K by metric."""

    def __init__(
        self,
        save_dir: str | Path,
        keep_top_k: int = 2,
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.metric = metric
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(
        self,
        state: dict[str, Any],
        metric_value: float,
        step: int,
    ) -> Path:
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))

        # Sort by metric
        self.history.sort(
            key=lambda x: x[0],
            reverse=(self.mode == "max"),
        )

        # Remove excess checkpoints
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)

        # Copy best
        best_val, best_path = self.history[0]
        best_dest = self.save_dir / "best.pth"
        shutil.copy2(best_path, best_dest)

        return path


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping monitor."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ---------------------------------------------------------------------------
# OC-SVM fitting
# ---------------------------------------------------------------------------

def fit_ocsvms(
    model: SecureLinkModel,
    dataloader: DataLoader,
    num_uavs: int,
    device: torch.device,
    kernel: str = "rbf",
    nu: float = 0.1,
    gamma: str = "scale",
) -> dict[int, Any]:
    """Fit one-class SVMs per registered UAV using training embeddings.

    Args:
        model: Trained SecureLink DNN
        dataloader: Training data loader
        num_uavs: Number of registered UAVs
        device: Compute device
        kernel: SVM kernel type
        nu: SVM nu parameter
        gamma: SVM gamma parameter

    Returns:
        Dict mapping UAV ID -> fitted OC-SVM
    """
    from sklearn.svm import OneClassSVM

    model.eval()
    embeddings_by_uav: dict[int, list] = {i: [] for i in range(num_uavs)}

    with torch.no_grad():
        for batch in dataloader:
            csi = batch["csi"].to(device)
            mems = batch["mems"].to(device)
            labels = batch["label"]

            emb = model(csi, mems).cpu().numpy()

            for i, label in enumerate(labels.numpy()):
                if label < num_uavs:
                    embeddings_by_uav[int(label)].append(emb[i])

    ocsvms = {}
    for uav_id in range(num_uavs):
        embs = embeddings_by_uav[uav_id]
        if len(embs) < 2:
            print(f"[WARN] UAV {uav_id} has {len(embs)} samples, skipping OC-SVM")
            continue

        X = np.stack(embs, axis=0)
        svm = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        svm.fit(X)
        ocsvms[uav_id] = svm
        print(f"[OC-SVM] UAV {uav_id}: fitted on {len(embs)} embeddings")

    return ocsvms


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    config: SecureLinkConfig,
    resume: str | None = None,
    max_steps: int | None = None,
) -> None:
    """Run the full SecureLink training pipeline.

    Args:
        config: Full configuration
        resume: Path to checkpoint to resume from
        max_steps: Maximum training steps (for smoke tests)
    """
    # Setup
    seed_everything(config.training.seed)
    device = get_device()
    ensure_artifact_dirs()

    print(f"[CONFIG] Training SecureLink on {device}")
    print(f"[CONFIG] Batch size: {config.training.batch_size}")
    print(f"[CONFIG] Epochs: {config.training.epochs}")
    print(f"[CONFIG] Learning rate: {config.training.learning_rate}")

    # Load data
    data_root = Path(config.data.data_root)
    if not data_root.is_absolute():
        # Relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_root = project_root / data_root

    print(f"[DATA] Loading from {data_root}")
    csi_all, mems_all, labels_all = load_all_data(data_root, config.data)

    print(f"[DATA] Total samples: {len(labels_all)}")
    print(f"[DATA] CSI shape: {csi_all.shape}, MEMS shape: {mems_all.shape}")

    # Determine actual CSI feature dimension from data
    csi_features = csi_all.shape[-1]
    mems_features = mems_all.shape[-1]

    # Split
    datasets = split_dataset_a(
        csi_all, mems_all, labels_all,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        seed=config.training.seed,
    )

    for name, ds in datasets.items():
        print(f"[DATA] {name}: {len(ds)} samples")

    batch_size = (
        config.training.batch_size
        if isinstance(config.training.batch_size, int)
        else 256
    )
    loaders = build_dataloaders(
        datasets,
        batch_size=batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # Build model
    model = build_model(config.model, csi_features=csi_features, mems_features=mems_features)
    model = model.to(device)
    print(f"[MODEL] Parameters: {model.count_parameters():,}")

    # Loss, optimizer, scheduler
    criterion = MultiSimilarityLoss(
        alpha=config.loss.alpha,
        beta=config.loss.beta,
        margin=config.loss.margin,
        epsilon=config.loss.epsilon,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    total_steps = len(loaders["train"]) * config.training.epochs
    warmup_steps = int(total_steps * config.training.warmup_ratio)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps, total_steps, min_lr=config.training.min_lr
    )

    # Checkpoint and early stopping
    ckpt_manager = CheckpointManager(
        save_dir=config.checkpoint.output_dir,
        keep_top_k=config.checkpoint.keep_top_k,
        metric=config.checkpoint.metric,
        mode=config.checkpoint.mode,
    )
    early_stop = EarlyStopping(
        patience=config.early_stopping.patience,
        min_delta=config.early_stopping.min_delta,
        mode=config.checkpoint.mode,
    )

    # Mixed precision
    use_amp = config.training.precision in ("fp16", "bf16") and device.type == "cuda"
    scaler = GradScaler(enabled=(config.training.precision == "fp16"))
    amp_dtype = torch.bfloat16 if config.training.precision == "bf16" else torch.float16

    # TensorBoard
    writer = SummaryWriter(log_dir=config.logging.tensorboard_dir)

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        print(f"[RESUME] Loading checkpoint from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("global_step", 0)

    # Training loop
    print(f"[TRAIN] Starting from epoch {start_epoch}")
    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in loaders["train"]:
            csi = batch["csi"].to(device)
            mems = batch["mems"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type="cuda", dtype=amp_dtype):
                    embeddings = model(csi, mems)
                    loss = criterion(embeddings, labels)

                if config.training.precision == "fp16":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    optimizer.step()
            else:
                embeddings = model(csi, mems)
                loss = criterion(embeddings, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()

            scheduler.step()
            global_step += 1

            # NaN detection
            if torch.isnan(loss):
                print("[FATAL] Loss is NaN -- stopping training")
                print("[FIX] Reduce lr by 10x, check data, check gradient clipping")
                return

            epoch_loss += loss.item()
            num_batches += 1

            # Max steps check (smoke test)
            if max_steps is not None and global_step >= max_steps:
                break

            # Step-based logging
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_lr(), global_step)

            # Step-based checkpointing
            if global_step % config.checkpoint.save_every_n_steps == 0:
                # Quick validation
                val_loss = _validate(
                    model, criterion, loaders.get("val"), device, use_amp, amp_dtype
                )
                ckpt_state = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": val_loss,
                    "config": config.model_dump(),
                }
                ckpt_manager.save(ckpt_state, val_loss, global_step)

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        val_loss = _validate(model, criterion, loaders.get("val"), device, use_amp, amp_dtype)
        elapsed = time.time() - t0

        print(
            f"[Epoch {epoch + 1}/{config.training.epochs}] "
            f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} "
            f"lr={scheduler.get_lr():.6f} time={elapsed:.1f}s"
        )

        writer.add_scalar("epoch/train_loss", avg_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)

        # Checkpoint
        ckpt_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "val_loss": val_loss,
            "config": config.model_dump(),
        }
        ckpt_manager.save(ckpt_state, val_loss, global_step)

        # Early stopping
        if config.early_stopping.enabled and early_stop.step(val_loss):
            print(f"[EARLY STOP] No improvement for {early_stop.patience} epochs. Stopping.")
            break

        # Max steps exit
        if max_steps is not None and global_step >= max_steps:
            print(f"[MAX STEPS] Reached {max_steps} steps. Stopping.")
            break

    writer.close()

    # Fit OC-SVMs after DNN training
    print("[OC-SVM] Fitting one-class SVMs per UAV...")
    ocsvms = fit_ocsvms(
        model, loaders["train"], config.data.num_uavs, device,
        kernel=config.ocsvm.kernel,
        nu=config.ocsvm.nu,
        gamma=config.ocsvm.gamma,
    )

    # Save OC-SVM models
    ocsvm_dir = Path(config.checkpoint.output_dir) / "ocsvm_models"
    ocsvm_dir.mkdir(parents=True, exist_ok=True)
    for uav_id, svm in ocsvms.items():
        ocsvm_path = ocsvm_dir / f"ocsvm_uav_{uav_id:02d}.pkl"
        with open(ocsvm_path, "wb") as f:
            pickle.dump(svm, f)
    print(f"[OC-SVM] Saved {len(ocsvms)} OC-SVM models to {ocsvm_dir}")

    print("[DONE] Training complete.")


def _validate(
    model: SecureLinkModel,
    criterion: nn.Module,
    dataloader: DataLoader | None,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Run validation and return average loss."""
    if dataloader is None:
        return float("inf")

    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            csi = batch["csi"].to(device)
            mems = batch["mems"].to(device)
            labels = batch["label"].to(device)

            if use_amp:
                with autocast(device_type="cuda", dtype=amp_dtype):
                    embeddings = model(csi, mems)
                    loss = criterion(embeddings, labels)
            else:
                embeddings = model(csi, mems)
                loss = criterion(embeddings, labels)

            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / max(num_batches, 1)
