#!/usr/bin/env python3
"""SecureLink training entry point.

Usage:
    python scripts/train.py --config configs/paper.toml
    python scripts/train.py --config configs/debug.toml --resume /path/to/checkpoint.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from securelink.train import train
from securelink.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureLink Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/paper.toml",
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (for smoke tests)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
