#!/usr/bin/env python3
"""SecureLink GPU batch size finder.

Finds optimal batch size for L4 GPUs (23GB VRAM).
Note: SecureLink model is very small (~1.3M params), so even batch_size=256
uses <500MB VRAM. The bottleneck is compute, not memory.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/find_batch_size.py --target 0.65
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from securelink.model import build_model
from securelink.utils import ModelConfig


def find_optimal_batch(target_util: float = 0.65) -> int:
    """Find optimal batch size for current GPU."""
    if not torch.cuda.is_available():
        print("[BATCH] No GPU available, defaulting to batch_size=32")
        return 32

    device = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(0).total_mem
    print(f"[BATCH] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[BATCH] Total VRAM: {total_mem / 1e6:.0f}MB")
    print(f"[BATCH] Target utilization: {target_util * 100:.0f}%")

    config = ModelConfig()
    model = build_model(config, csi_features=52, mems_features=8).to(device)

    # Binary search for batch size
    low, high = 32, 8192
    best_bs = 256  # paper default

    while low <= high:
        mid = (low + high) // 2
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        try:
            csi = torch.randn(mid, 6, 52, device=device)
            mems = torch.randn(mid, 6, 8, device=device)
            labels = torch.randint(0, 22, (mid,), device=device)

            emb = model(csi, mems)
            loss = (emb * emb).sum()
            loss.backward()

            peak = torch.cuda.max_memory_allocated()
            util = peak / total_mem

            if util < target_util:
                best_bs = mid
                low = mid + 1
            else:
                high = mid - 1

            del csi, mems, labels, emb, loss
            model.zero_grad()
            torch.cuda.empty_cache()

        except RuntimeError:
            high = mid - 1
            torch.cuda.empty_cache()

    print(f"[BATCH] Optimal batch_size={best_bs}")
    print(f"[BATCH] Note: SecureLink is a tiny model (<5MB). VRAM is not the bottleneck.")
    return best_bs


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureLink Batch Size Finder")
    parser.add_argument("--target", type=float, default=0.65)
    args = parser.parse_args()

    bs = find_optimal_batch(args.target)
    print(f"\nRecommended: batch_size={bs}")


if __name__ == "__main__":
    main()
