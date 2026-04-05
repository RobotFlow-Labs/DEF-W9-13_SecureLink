#!/usr/bin/env python3
"""SecureLink evaluation entry point.

Usage:
    python scripts/evaluate.py --config configs/paper.toml \
        --checkpoint /mnt/artifacts-datai/checkpoints/securelink/best.pth \
        --ocsvm-dir /mnt/artifacts-datai/checkpoints/securelink/ocsvm_models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from securelink.dataset import (
    build_dataloaders,
    load_all_data,
    split_dataset_a,
    split_dataset_b,
)
from securelink.evaluate import (
    evaluate_closed_world,
    evaluate_open_world,
    load_ocsvms,
    save_report,
)
from securelink.model import build_model
from securelink.utils import (
    ModelConfig,
    ensure_artifact_dirs,
    get_device,
    load_config,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureLink Evaluation")
    parser.add_argument("--config", type=str, default="configs/paper.toml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ocsvm-dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["closed", "open", "both"], default="both"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.training.seed)
    device = get_device()
    dirs = ensure_artifact_dirs()

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})
    model_config = ModelConfig(**config_dict.get("model", config.model.model_dump()))

    # Load data to get feature dimensions
    data_root = Path(config.data.data_root)
    if not data_root.is_absolute():
        project_root = Path(__file__).parent.parent
        data_root = project_root / data_root

    csi_all, mems_all, labels_all = load_all_data(data_root, config.data)
    csi_features = csi_all.shape[-1]
    mems_features = mems_all.shape[-1]

    model = build_model(model_config, csi_features=csi_features, mems_features=mems_features)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Load OC-SVMs
    ocsvms = load_ocsvms(args.ocsvm_dir)

    if args.mode in ("closed", "both"):
        print("\n=== Closed-World Evaluation (Dataset A) ===")
        datasets_a = split_dataset_a(
            csi_all, mems_all, labels_all,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.training.seed,
        )
        loaders_a = build_dataloaders(
            {"test": datasets_a["test"]},
            batch_size=256,
            num_workers=0,
        )
        metrics_a = evaluate_closed_world(model, loaders_a["test"], ocsvms, device)
        print(f"  Accuracy: {metrics_a['accuracy']:.4f}")
        print(f"  Recall:   {metrics_a['recall']:.4f}")
        print(f"  Precision:{metrics_a['precision']:.4f}")
        save_report(metrics_a, dirs["reports"] / "closed_world_report.json")

    if args.mode in ("open", "both"):
        print("\n=== Open-World Evaluation (Dataset B) ===")
        datasets_b = split_dataset_b(
            csi_all, mems_all, labels_all,
            seed=config.training.seed,
        )
        loaders_b = build_dataloaders(
            {"test": datasets_b["test"]},
            batch_size=256,
            num_workers=0,
        )
        metrics_b = evaluate_open_world(model, loaders_b["test"], ocsvms, device)
        print(f"  Accuracy: {metrics_b['accuracy']:.4f}")
        print(f"  TNR:      {metrics_b['tnr']:.4f}")
        print(f"  Recall:   {metrics_b['recall']:.4f}")
        print(f"  Precision:{metrics_b['precision']:.4f}")
        save_report(metrics_b, dirs["reports"] / "open_world_report.json")


if __name__ == "__main__":
    main()
