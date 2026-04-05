"""Utilities: configuration loading, seeding, device selection."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class DataConfig(BaseModel):
    """Dataset configuration."""

    data_root: str = "repositories/SecureLink_data"
    sample_length: int = Field(6, description="Number of frames per sample (M)")
    csi_subcarriers: int = Field(52, description="Number of CSI subcarriers")
    mems_fields: list[str] = Field(
        default=["pitch", "roll", "yaw", "tof", "baro", "agx", "agy", "agz"],
        description="MEMS telemetry fields to use",
    )
    phase_variance_threshold: float = Field(
        4.0, description="Eta threshold for CSI outlier filtering"
    )
    num_uavs: int = Field(22, description="Total number of UAVs in dataset")
    num_registered: int = Field(20, description="Number of registered UAVs (Dataset B)")
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    cnn_filters: int = Field(64, description="Number of CNN filters per layer")
    cnn_kernel_sizes: list[int] = Field([3, 2], description="Kernel sizes for 2 conv layers")
    lstm_hidden: int = Field(64, description="BiLSTM hidden size per direction")
    lstm_layers: int = Field(1, description="Number of BiLSTM layers")
    attention_heads: int = Field(4, description="Number of attention heads")
    attention_layers: int = Field(2, description="Number of attention layers")
    embedding_dim: int = Field(256, description="Final embedding dimension")
    ff_dim: int = Field(512, description="Feed-forward hidden dimension in attention")
    dropout: float = Field(0.1, description="Dropout rate")


class LossConfig(BaseModel):
    """Loss function configuration."""

    alpha: float = Field(1.0, description="Multi-similarity loss alpha (positive scaling)")
    beta: float = Field(10.0, description="Multi-similarity loss beta (negative scaling)")
    margin: float = Field(0.5, description="Multi-similarity loss margin mu")
    epsilon: float = Field(0.1, description="Threshold for pair mining")


class TrainingConfig(BaseModel):
    """Training loop configuration."""

    batch_size: int | str = Field(256, description="Batch size or 'auto'")
    learning_rate: float = 0.001
    epochs: int = 30
    optimizer: str = "adam"
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    min_lr: float = 1e-6
    precision: str = "bf16"
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42


class CheckpointConfig(BaseModel):
    """Checkpoint configuration."""

    output_dir: str = "/mnt/artifacts-datai/checkpoints/securelink"
    save_every_n_steps: int = 500
    keep_top_k: int = 2
    metric: str = "val_loss"
    mode: str = "min"


class EarlyStoppingConfig(BaseModel):
    """Early stopping configuration."""

    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = "/mnt/artifacts-datai/logs/securelink"
    tensorboard_dir: str = "/mnt/artifacts-datai/tensorboard/securelink"


class OCSVMConfig(BaseModel):
    """OC-SVM configuration."""

    kernel: str = "rbf"
    nu: float = 0.1
    gamma: str = "scale"


class SecureLinkConfig(BaseSettings):
    """Root configuration for SecureLink."""

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    logging: LoggingConfig = LoggingConfig()
    ocsvm: OCSVMConfig = OCSVMConfig()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> SecureLinkConfig:
    """Load configuration from a TOML file."""
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return SecureLinkConfig(**raw)


def config_to_dict(config: SecureLinkConfig) -> dict[str, Any]:
    """Convert config to a flat dictionary for logging."""
    return config.model_dump()


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """Select compute device.

    Args:
        preference: 'auto', 'cuda', 'cpu', or 'mps'

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Artifact directory helpers
# ---------------------------------------------------------------------------

PROJECT_NAME = "securelink"
ARTIFACTS_ROOT = Path("/mnt/artifacts-datai")


def ensure_artifact_dirs() -> dict[str, Path]:
    """Create and return artifact directory paths."""
    dirs = {
        "checkpoints": ARTIFACTS_ROOT / "checkpoints" / PROJECT_NAME,
        "models": ARTIFACTS_ROOT / "models" / PROJECT_NAME,
        "logs": ARTIFACTS_ROOT / "logs" / PROJECT_NAME,
        "exports": ARTIFACTS_ROOT / "exports" / PROJECT_NAME,
        "reports": ARTIFACTS_ROOT / "reports" / PROJECT_NAME,
        "tensorboard": ARTIFACTS_ROOT / "tensorboard" / PROJECT_NAME,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
