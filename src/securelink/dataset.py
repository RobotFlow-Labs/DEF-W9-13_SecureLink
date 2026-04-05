"""SecureLink dataset: CSI + MEMS fingerprint loading, alignment, and PyTorch Dataset.

Data flow:
  1. Parse PicoScenes .csi binary files -> extract CSI phase errors
  2. Load telemetry .csv files -> select 8 MEMS fields
  3. Filter CSI outliers by phase gradient variance (eta=4)
  4. Clean telemetry outlier frames
  5. Align CSI to telemetry via linear interpolation (L << M)
  6. Construct samples of M=6 consecutive aligned frames
  7. Split into train/val/test for Dataset A (closed-world) or Dataset B (open-world)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset

from securelink.utils import DataConfig

# ---------------------------------------------------------------------------
# CSI Parsing (PicoScenes binary format)
# ---------------------------------------------------------------------------

def parse_csi_file(filepath: str | Path) -> np.ndarray | None:
    """Parse a PicoScenes .csi binary file and extract CSI phase information.

    The PicoScenes format stores CSI measurements as complex values across
    subcarriers. We extract the phase from each measurement.

    Args:
        filepath: Path to .csi binary file

    Returns:
        Array of shape (num_frames, num_subcarriers) containing phase values,
        or None if parsing fails.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    try:
        data = filepath.read_bytes()
    except OSError:
        return None

    if len(data) < 100:
        return None

    # PicoScenes CSI files contain multiple frames.
    # Each frame has a header followed by CSI complex values.
    # We search for "RxSBasic" markers to find frame boundaries.
    frames = []
    marker = b"RxSBasic"
    offset = 0

    while True:
        idx = data.find(marker, offset)
        if idx == -1:
            break

        # Try to extract CSI data after the marker
        # PicoScenes stores CSI as pairs of int16 (I, Q) values
        # for each subcarrier. 52 subcarriers = 104 int16 values.
        csi_offset = idx + len(marker) + 32  # Skip header fields
        num_subcarriers = 52
        csi_bytes = num_subcarriers * 4  # 2 int16 per subcarrier

        if csi_offset + csi_bytes > len(data):
            offset = idx + 1
            continue

        try:
            iq_data = struct.unpack(
                f"<{num_subcarriers * 2}h",
                data[csi_offset : csi_offset + csi_bytes],
            )
            # Reconstruct complex CSI values
            i_vals = np.array(iq_data[0::2], dtype=np.float64)
            q_vals = np.array(iq_data[1::2], dtype=np.float64)
            phase = np.arctan2(q_vals, i_vals)
            frames.append(phase)
        except (struct.error, ValueError):
            pass

        offset = idx + 1

    if len(frames) == 0:
        return None

    return np.stack(frames, axis=0)  # (num_frames, 52)


def extract_phase_errors(phases: np.ndarray, eta: float = 4.0) -> np.ndarray:
    """Extract CSI phase errors from raw phase measurements.

    Phase error: e = phi - psi - 2*pi*lambda*i - z
    We estimate lambda and z from the phase differences across subcarriers,
    then compute the residual phase error.

    Also filters outlier frames by phase gradient variance threshold.

    Args:
        phases: (num_frames, K) raw phase values
        eta: Variance threshold for outlier filtering

    Returns:
        (filtered_frames, K) phase errors with outliers removed
    """
    if phases.shape[0] == 0:
        return phases

    num_frames, num_subs = phases.shape

    # Remove DC (zero) subcarrier contribution by centering
    # Compute phase gradient across subcarriers
    phase_grad = np.diff(phases, axis=1)  # (num_frames, K-1)

    # Filter frames with high gradient variance (environmental interference)
    grad_var = np.var(phase_grad, axis=1)  # (num_frames,)
    valid_mask = grad_var < eta
    filtered_phases = phases[valid_mask]

    if filtered_phases.shape[0] == 0:
        # If all frames filtered, relax threshold
        return phases

    # Estimate linear phase components (lambda, z) using least squares
    # Subcarrier indices
    subcarrier_idx = np.arange(num_subs)

    phase_errors = []
    for frame in filtered_phases:
        # Fit linear model: phase = lambda * i + z + error
        # Using simple least squares
        A = np.column_stack([subcarrier_idx, np.ones(num_subs)])
        result = np.linalg.lstsq(A, frame, rcond=None)
        linear_fit = A @ result[0]
        error = frame - linear_fit
        phase_errors.append(error)

    return np.stack(phase_errors, axis=0)


# ---------------------------------------------------------------------------
# Telemetry Loading
# ---------------------------------------------------------------------------

MEMS_FIELDS = ["pitch", "roll", "yaw", "tof", "baro", "agx", "agy", "agz"]


def load_telemetry(filepath: str | Path, fields: list[str] | None = None) -> np.ndarray:
    """Load telemetry CSV and extract selected MEMS fields.

    Args:
        filepath: Path to telemetry CSV file
        fields: List of field names to extract (default: 8 MEMS fields)

    Returns:
        (num_frames, num_fields) array of telemetry values
    """
    if fields is None:
        fields = MEMS_FIELDS

    filepath = Path(filepath)
    df = pd.read_csv(filepath)

    # Select only the desired fields
    available = [f for f in fields if f in df.columns]
    if len(available) == 0:
        raise ValueError(f"No matching fields in {filepath}. Columns: {list(df.columns)}")

    data = df[available].values.astype(np.float32)
    return data


def clean_telemetry(data: np.ndarray, max_std: float = 5.0) -> np.ndarray:
    """Clean telemetry frames by removing outliers.

    Outlier detection: frames where any field exceeds max_std standard deviations
    from the field mean are removed.

    Args:
        data: (num_frames, num_fields)
        max_std: Maximum standard deviations before marking as outlier

    Returns:
        Cleaned data with outlier frames removed
    """
    if data.shape[0] < 3:
        return data

    means = np.mean(data, axis=0, keepdims=True)
    stds = np.std(data, axis=0, keepdims=True)
    stds = np.maximum(stds, 1e-8)  # Avoid division by zero

    z_scores = np.abs((data - means) / stds)
    valid_mask = np.all(z_scores < max_std, axis=1)

    cleaned = data[valid_mask]
    if cleaned.shape[0] == 0:
        return data  # Don't return empty array

    return cleaned


# ---------------------------------------------------------------------------
# Data Alignment
# ---------------------------------------------------------------------------

def align_csi_to_telemetry(
    csi: np.ndarray,
    telemetry: np.ndarray,
) -> np.ndarray:
    """Align CSI frames to telemetry frames via linear interpolation.

    CSI measurements are much sparser than telemetry (L << M).
    We upsample CSI to match the telemetry frame count M using
    linear interpolation.

    Args:
        csi: (L, K) CSI phase errors, L frames
        telemetry: (M, F) telemetry data, M frames

    Returns:
        (M, K) interpolated CSI data aligned to telemetry timestamps
    """
    L, K = csi.shape
    M = telemetry.shape[0]

    if L >= M:
        # CSI has more or equal frames -- subsample to match
        indices = np.linspace(0, L - 1, M).astype(int)
        return csi[indices]

    if L < 2:
        # Not enough CSI frames for interpolation -- repeat
        return np.tile(csi, (M, 1))[:M]

    # Linear interpolation: create M evenly-spaced CSI frames from L original ones
    src_indices = np.linspace(0, 1, L)
    dst_indices = np.linspace(0, 1, M)

    interpolator = interp1d(src_indices, csi, axis=0, kind="linear", fill_value="extrapolate")
    aligned = interpolator(dst_indices)

    return aligned.astype(np.float32)


# ---------------------------------------------------------------------------
# Sample Construction
# ---------------------------------------------------------------------------

def construct_samples(
    csi: np.ndarray,
    telemetry: np.ndarray,
    sample_length: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct fingerprint samples from aligned CSI and telemetry sequences.

    Each sample contains M consecutive frames from both modalities.

    Args:
        csi: (N, K) aligned CSI phase errors
        telemetry: (N, F) telemetry data
        sample_length: M, number of frames per sample

    Returns:
        csi_samples: (num_samples, M, K)
        mems_samples: (num_samples, M, F)
    """
    N = min(csi.shape[0], telemetry.shape[0])
    if N < sample_length:
        # Pad if not enough frames
        pad_csi = np.pad(csi, ((0, sample_length - N), (0, 0)), mode="edge")
        pad_mems = np.pad(telemetry, ((0, sample_length - N), (0, 0)), mode="edge")
        return pad_csi[np.newaxis], pad_mems[np.newaxis]

    num_samples = N // sample_length
    csi_samples = []
    mems_samples = []

    for i in range(num_samples):
        start = i * sample_length
        end = start + sample_length
        csi_samples.append(csi[start:end])
        mems_samples.append(telemetry[start:end])

    return np.stack(csi_samples), np.stack(mems_samples)


# ---------------------------------------------------------------------------
# Trial Loading (single UAV, single trial)
# ---------------------------------------------------------------------------

def load_trial(
    data_root: Path,
    uav_id: int,
    trial_id: int,
    mems_fields: list[str] | None = None,
    eta: float = 4.0,
    sample_length: int = 6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load and process a single trial for one UAV.

    Args:
        data_root: Root directory of SecureLink_data
        uav_id: UAV index (0-21)
        trial_id: Trial index (1-25)
        mems_fields: MEMS fields to extract
        eta: Phase variance threshold
        sample_length: Frames per sample (M)

    Returns:
        (csi_samples, mems_samples) or None if loading fails
    """
    csi_dir = data_root / f"CSI_{uav_id:02d}"
    sensor_dir = data_root / f"sensors_{uav_id:02d}"

    csi_file = csi_dir / f"{trial_id}.csi"
    sensor_file = sensor_dir / f"{trial_id}.csv"

    if not csi_file.exists() or not sensor_file.exists():
        return None

    # Load and process CSI
    raw_phases = parse_csi_file(csi_file)
    if raw_phases is None or raw_phases.shape[0] == 0:
        return None

    phase_errors = extract_phase_errors(raw_phases, eta=eta)
    if phase_errors.shape[0] == 0:
        return None

    # Load and process telemetry
    telemetry = load_telemetry(sensor_file, fields=mems_fields)
    telemetry = clean_telemetry(telemetry)
    if telemetry.shape[0] == 0:
        return None

    # Align CSI to telemetry
    aligned_csi = align_csi_to_telemetry(phase_errors, telemetry)

    # Construct samples
    csi_samples, mems_samples = construct_samples(aligned_csi, telemetry, sample_length)

    return csi_samples, mems_samples


# ---------------------------------------------------------------------------
# Full Dataset Loading
# ---------------------------------------------------------------------------

def load_all_data(
    data_root: str | Path,
    config: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all UAV data from all trials.

    Args:
        data_root: Path to SecureLink_data directory
        config: Data configuration

    Returns:
        csi_all: (total_samples, M, K)
        mems_all: (total_samples, M, F)
        labels: (total_samples,) UAV IDs
    """
    data_root = Path(data_root)
    all_csi = []
    all_mems = []
    all_labels = []

    for uav_id in range(config.num_uavs):
        for trial_id in range(1, 26):  # Trials 1-25
            result = load_trial(
                data_root=data_root,
                uav_id=uav_id,
                trial_id=trial_id,
                mems_fields=config.mems_fields,
                eta=config.phase_variance_threshold,
                sample_length=config.sample_length,
            )
            if result is None:
                continue

            csi_samples, mems_samples = result
            n = csi_samples.shape[0]
            all_csi.append(csi_samples)
            all_mems.append(mems_samples)
            all_labels.append(np.full(n, uav_id, dtype=np.int64))

    if not all_csi:
        raise ValueError(f"No data loaded from {data_root}")

    return (
        np.concatenate(all_csi, axis=0),
        np.concatenate(all_mems, axis=0),
        np.concatenate(all_labels, axis=0),
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SecureLinkDataset(Dataset):
    """PyTorch dataset for SecureLink CSI + MEMS fingerprint samples.

    Each item returns:
        csi: (M, K) float32 tensor of CSI phase errors
        mems: (M, F) float32 tensor of MEMS telemetry
        label: int64 scalar UAV ID
    """

    def __init__(
        self,
        csi: np.ndarray,
        mems: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
    ):
        self.csi = torch.from_numpy(csi.astype(np.float32))
        self.mems = torch.from_numpy(mems.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.normalize = normalize

        if normalize:
            # Per-feature z-score normalization
            self._csi_mean = self.csi.mean(dim=(0, 1), keepdim=True)
            self._csi_std = self.csi.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)
            self._mems_mean = self.mems.mean(dim=(0, 1), keepdim=True)
            self._mems_std = self.mems.std(dim=(0, 1), keepdim=True).clamp(min=1e-8)

            self.csi = (self.csi - self._csi_mean) / self._csi_std
            self.mems = (self.mems - self._mems_mean) / self._mems_std

    def __len__(self) -> int:
        return self.csi.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "csi": self.csi[idx],
            "mems": self.mems[idx],
            "label": self.labels[idx],
        }


# ---------------------------------------------------------------------------
# Dataset Splitting
# ---------------------------------------------------------------------------

def split_dataset_a(
    csi: np.ndarray,
    mems: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, SecureLinkDataset]:
    """Create Dataset A (closed-world) with train/val/test split.

    All 22 UAVs are included. Each UAV's samples are shuffled and split
    60/20/20 for train/validation/test.

    Args:
        csi: (N, M, K) all CSI samples
        mems: (N, M, F) all MEMS samples
        labels: (N,) UAV IDs

    Returns:
        Dict with 'train', 'val', 'test' SecureLinkDataset instances
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return {
        "train": SecureLinkDataset(csi[train_idx], mems[train_idx], labels[train_idx]),
        "val": SecureLinkDataset(csi[val_idx], mems[val_idx], labels[val_idx]),
        "test": SecureLinkDataset(csi[test_idx], mems[test_idx], labels[test_idx]),
    }


def split_dataset_b(
    csi: np.ndarray,
    mems: np.ndarray,
    labels: np.ndarray,
    registered_ids: list[int] | None = None,
    impersonator_ids: list[int] | None = None,
    seed: int = 42,
) -> dict[str, SecureLinkDataset]:
    """Create Dataset B (open-world) with registered and impersonating UAVs.

    20 UAVs are registered (positive). 2 UAVs are impersonators whose samples
    are randomly assigned registered IDs. 80% train+val, 20% test.

    Args:
        csi: (N, M, K) all CSI samples
        mems: (N, M, F) all MEMS samples
        labels: (N,) true UAV IDs
        registered_ids: List of registered UAV IDs (default: 0-19)
        impersonator_ids: List of impersonating UAV IDs (default: 20, 21)

    Returns:
        Dict with 'train', 'test' SecureLinkDataset instances
    """
    if registered_ids is None:
        registered_ids = list(range(20))
    if impersonator_ids is None:
        impersonator_ids = [20, 21]

    rng = np.random.RandomState(seed)

    # Registered UAV data
    reg_mask = np.isin(labels, registered_ids)
    reg_csi = csi[reg_mask]
    reg_mems = mems[reg_mask]
    reg_labels = labels[reg_mask]

    # Impersonator data -- assign random registered IDs
    imp_mask = np.isin(labels, impersonator_ids)
    imp_csi = csi[imp_mask]
    imp_mems = mems[imp_mask]
    imp_labels_true = labels[imp_mask]
    # Assign random registered IDs to impersonators
    imp_labels_fake = rng.choice(registered_ids, size=len(imp_labels_true))

    # Split registered: 80% train, 20% test
    reg_indices = np.arange(len(reg_labels))
    rng.shuffle(reg_indices)
    n_train = int(len(reg_indices) * 0.8)
    train_idx = reg_indices[:n_train]
    test_reg_idx = reg_indices[n_train:]

    # Test set: remaining registered + all impersonators
    test_csi = np.concatenate([reg_csi[test_reg_idx], imp_csi], axis=0)
    test_mems = np.concatenate([reg_mems[test_reg_idx], imp_mems], axis=0)
    test_labels = np.concatenate(
        [reg_labels[test_reg_idx], imp_labels_fake.astype(np.int64)], axis=0
    )

    return {
        "train": SecureLinkDataset(reg_csi[train_idx], reg_mems[train_idx], reg_labels[train_idx]),
        "test": SecureLinkDataset(test_csi, test_mems, test_labels),
    }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    datasets: dict[str, SecureLinkDataset],
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> dict[str, DataLoader]:
    """Build DataLoaders from dataset dict.

    Args:
        datasets: Dict of split name -> SecureLinkDataset
        batch_size: Batch size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer

    Returns:
        Dict of split name -> DataLoader
    """
    loaders = {}
    for name, ds in datasets.items():
        shuffle = name == "train"
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(name == "train"),
        )
    return loaders
