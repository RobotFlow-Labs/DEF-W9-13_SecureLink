"""SecureLink evaluation: closed-world and open-world authentication metrics.

Evaluation protocol:
  - Closed-world (Dataset A): All 22 UAVs are known. For each test sample,
    run through all OC-SVMs and pick the one with highest score.
  - Open-world (Dataset B): 20 registered + 2 impersonating. OC-SVMs must
    reject impersonators (negative OC-SVM decision).

Metrics: Accuracy, TNR (True Negative Rate), Recall, Precision
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from securelink.model import SecureLinkModel


def extract_embeddings(
    model: SecureLinkModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings and labels from a dataloader.

    Args:
        model: Trained SecureLink model
        dataloader: Data loader
        device: Compute device

    Returns:
        embeddings: (N, embedding_dim) numpy array
        labels: (N,) numpy array of true UAV IDs
    """
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            csi = batch["csi"].to(device)
            mems = batch["mems"].to(device)
            labels = batch["label"]

            emb = model(csi, mems).cpu().numpy()
            all_embeddings.append(emb)
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings), np.concatenate(all_labels)


def predict_with_ocsvms(
    embeddings: np.ndarray,
    ocsvms: dict[int, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Predict UAV identity using OC-SVMs.

    For each embedding, run all OC-SVMs and select the one with the highest
    decision function score. If all OC-SVMs reject, mark as unknown (-1).

    Args:
        embeddings: (N, D) embedding vectors
        ocsvms: Dict mapping UAV ID -> fitted OC-SVM

    Returns:
        predicted_ids: (N,) predicted UAV IDs (-1 for unknown)
        confidence: (N,) highest OC-SVM decision scores
    """
    N = embeddings.shape[0]
    uav_ids = sorted(ocsvms.keys())
    num_uavs = len(uav_ids)

    # Score matrix: (N, num_uavs)
    scores = np.full((N, num_uavs), -np.inf)

    for j, uav_id in enumerate(uav_ids):
        svm = ocsvms[uav_id]
        scores[:, j] = svm.decision_function(embeddings)

    # For each sample, pick the UAV with highest score
    best_idx = np.argmax(scores, axis=1)
    predicted_ids = np.array([uav_ids[idx] for idx in best_idx])
    confidence = scores[np.arange(N), best_idx]

    # If best score is negative, OC-SVM rejects -> unknown
    rejected = confidence < 0
    predicted_ids[rejected] = -1

    return predicted_ids, confidence


def compute_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    num_classes: int = 22,
) -> dict[str, float]:
    """Compute authentication metrics.

    Args:
        true_labels: (N,) true UAV IDs
        predicted_labels: (N,) predicted UAV IDs (-1 for rejected)
        num_classes: Number of UAV classes

    Returns:
        Dict with accuracy, tnr, recall, precision
    """
    accuracy = accuracy_score(true_labels, predicted_labels)

    # For TNR: unauthorized (impersonating) samples correctly rejected
    # We treat this as a binary problem: authorized (correct ID) vs unauthorized
    is_correct = true_labels == predicted_labels
    is_rejected = predicted_labels == -1

    # TNR = proportion of unauthorized samples rejected
    # In closed-world, there are no unauthorized, so TNR is 1 by convention
    # In open-world, unauthorized = impersonator samples

    # Per-class recall and precision (macro average)
    valid_mask = predicted_labels >= 0
    if valid_mask.sum() > 0:
        recall = recall_score(
            true_labels[valid_mask],
            predicted_labels[valid_mask],
            average="macro",
            zero_division=0,
        )
        precision = precision_score(
            true_labels[valid_mask],
            predicted_labels[valid_mask],
            average="macro",
            zero_division=0,
        )
    else:
        recall = 0.0
        precision = 0.0

    return {
        "accuracy": float(accuracy),
        "recall": float(recall),
        "precision": float(precision),
        "num_samples": int(len(true_labels)),
        "num_correct": int(is_correct.sum()),
        "num_rejected": int(is_rejected.sum()),
    }


def compute_tnr(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    impersonator_ids: list[int] | None = None,
) -> float:
    """Compute True Negative Rate for impersonator detection.

    TNR = number of impersonators correctly rejected / total impersonators

    Args:
        true_labels: True UAV IDs
        predicted_labels: Predicted IDs (-1 for rejected)
        impersonator_ids: IDs of impersonating UAVs

    Returns:
        TNR value
    """
    if impersonator_ids is None:
        impersonator_ids = [20, 21]

    imp_mask = np.isin(true_labels, impersonator_ids)
    if imp_mask.sum() == 0:
        return 1.0  # No impersonators to detect

    # Correctly rejected: predicted as -1 or wrong ID
    imp_predictions = predicted_labels[imp_mask]
    correctly_rejected = (imp_predictions == -1).sum()

    return float(correctly_rejected / imp_mask.sum())


def evaluate_closed_world(
    model: SecureLinkModel,
    dataloader: DataLoader,
    ocsvms: dict[int, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run closed-world evaluation (Dataset A).

    Args:
        model: Trained model
        dataloader: Test data loader
        ocsvms: Fitted OC-SVMs
        device: Compute device

    Returns:
        Dict with metrics and confusion matrix
    """
    embeddings, true_labels = extract_embeddings(model, dataloader, device)
    predicted_labels, confidence = predict_with_ocsvms(embeddings, ocsvms)

    metrics = compute_metrics(true_labels, predicted_labels)

    # Confusion matrix
    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(22)),
    )

    metrics["confusion_matrix"] = cm.tolist()
    metrics["setting"] = "closed-world"

    return metrics


def evaluate_open_world(
    model: SecureLinkModel,
    dataloader: DataLoader,
    ocsvms: dict[int, Any],
    device: torch.device,
    impersonator_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Run open-world evaluation (Dataset B).

    Args:
        model: Trained model
        dataloader: Test data loader (includes impersonators)
        ocsvms: Fitted OC-SVMs (trained on registered UAVs only)
        device: Compute device
        impersonator_ids: IDs of impersonating UAVs

    Returns:
        Dict with metrics including TNR
    """
    if impersonator_ids is None:
        impersonator_ids = [20, 21]

    embeddings, true_labels = extract_embeddings(model, dataloader, device)
    predicted_labels, confidence = predict_with_ocsvms(embeddings, ocsvms)

    metrics = compute_metrics(true_labels, predicted_labels)
    tnr = compute_tnr(true_labels, predicted_labels, impersonator_ids)

    metrics["tnr"] = tnr
    metrics["setting"] = "open-world"

    return metrics


def save_report(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save evaluation report as JSON.

    Args:
        metrics: Evaluation metrics dict
        output_path: Path to save JSON report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = {k: convert(v) for k, v in metrics.items()}

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[REPORT] Saved to {output_path}")


def load_ocsvms(ocsvm_dir: str | Path) -> dict[int, Any]:
    """Load saved OC-SVM models.

    Args:
        ocsvm_dir: Directory containing ocsvm_uav_XX.pkl files

    Returns:
        Dict mapping UAV ID -> OC-SVM model
    """
    ocsvm_dir = Path(ocsvm_dir)
    ocsvms = {}

    for pkl_file in sorted(ocsvm_dir.glob("ocsvm_uav_*.pkl")):
        uav_id = int(pkl_file.stem.split("_")[-1])
        with open(pkl_file, "rb") as f:
            ocsvms[uav_id] = pickle.load(f)

    print(f"[OC-SVM] Loaded {len(ocsvms)} models from {ocsvm_dir}")
    return ocsvms
