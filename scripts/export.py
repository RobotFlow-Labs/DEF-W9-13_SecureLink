#!/usr/bin/env python3
"""SecureLink model export: pth -> safetensors -> ONNX -> TensorRT.

Usage:
    python scripts/export.py \
        --checkpoint /mnt/artifacts-datai/checkpoints/securelink/best.pth \
        --output-dir /mnt/artifacts-datai/exports/securelink/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from securelink.model import build_model
from securelink.utils import ModelConfig


def export_safetensors(model: torch.nn.Module, output_path: Path) -> None:
    """Export model to safetensors format."""
    from safetensors.torch import save_file

    state_dict = model.state_dict()
    save_file(state_dict, str(output_path))
    print(f"[EXPORT] safetensors: {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    sample_length: int = 6,
    csi_features: int = 52,
    mems_features: int = 8,
) -> None:
    """Export model to ONNX format."""
    model.eval()
    dummy_csi = torch.randn(1, sample_length, csi_features)
    dummy_mems = torch.randn(1, sample_length, mems_features)

    torch.onnx.export(
        model,
        (dummy_csi, dummy_mems),
        str(output_path),
        input_names=["csi", "mems"],
        output_names=["embedding"],
        dynamic_axes={
            "csi": {0: "batch"},
            "mems": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[EXPORT] ONNX: {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")

    # Verify
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("[EXPORT] ONNX model verified")


def export_tensorrt(
    onnx_path: Path,
    output_dir: Path,
    sample_length: int = 6,
    csi_features: int = 52,
    mems_features: int = 8,
) -> None:
    """Export ONNX model to TensorRT FP16 and FP32."""
    trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")

    if not trt_script.exists():
        print("[EXPORT] TRT toolkit not found, trying trtexec directly")
        import subprocess

        for precision in ["fp32", "fp16"]:
            out_path = output_dir / f"securelink_{precision}.engine"
            cmd = [
                "trtexec",
                f"--onnx={onnx_path}",
                f"--saveEngine={out_path}",
                f"--minShapes=csi:1x{sample_length}x{csi_features},"
                f"mems:1x{sample_length}x{mems_features}",
                f"--optShapes=csi:32x{sample_length}x{csi_features},"
                f"mems:32x{sample_length}x{mems_features}",
                f"--maxShapes=csi:256x{sample_length}x{csi_features},"
                f"mems:256x{sample_length}x{mems_features}",
            ]
            if precision == "fp16":
                cmd.append("--fp16")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(
                        f"[EXPORT] TRT {precision}: {out_path} "
                        f"({out_path.stat().st_size / 1e6:.1f}MB)"
                    )
                else:
                    print(f"[EXPORT] TRT {precision} FAILED: {result.stderr[:200]}")
            except FileNotFoundError:
                print("[EXPORT] trtexec not found — install TensorRT to export")
                return
            except subprocess.TimeoutExpired:
                print(f"[EXPORT] TRT {precision} timed out")
    else:
        import subprocess

        for precision in ["fp32", "fp16"]:
            out_path = output_dir / f"securelink_{precision}.engine"
            cmd = [
                sys.executable,
                str(trt_script),
                str(onnx_path),
                "--output", str(out_path),
                "--precision", precision,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"[EXPORT] TRT {precision}: {out_path}")
            else:
                print(f"[EXPORT] TRT {precision} FAILED: {result.stderr[:200]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SecureLink Model Export")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/artifacts-datai/exports/securelink",
    )
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config_dict = ckpt.get("config", {})
    model_config = ModelConfig(**config_dict.get("model", {}))

    # Determine feature dimensions from config
    csi_features = config_dict.get("data", {}).get("csi_subcarriers", 52)
    sample_length = config_dict.get("data", {}).get("sample_length", 6)
    mems_features = len(
        config_dict.get("data", {}).get(
            "mems_fields", ["pitch", "roll", "yaw", "tof", "baro", "agx", "agy", "agz"]
        )
    )

    model = build_model(model_config, csi_features=csi_features, mems_features=mems_features)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[EXPORT] Model: {model.count_parameters():,} parameters")
    print(
        f"[EXPORT] Input: CSI({sample_length}x{csi_features}), "
        f"MEMS({sample_length}x{mems_features})"
    )
    print(f"[EXPORT] Output dir: {output_dir}")

    # 1. Copy pth
    pth_path = output_dir / "securelink.pth"
    shutil.copy2(args.checkpoint, pth_path)
    print(f"[EXPORT] pth: {pth_path} ({pth_path.stat().st_size / 1e6:.1f}MB)")

    # 2. Safetensors
    safe_path = output_dir / "securelink.safetensors"
    export_safetensors(model, safe_path)

    # 3. ONNX
    onnx_path = output_dir / "securelink.onnx"
    export_onnx(model, onnx_path, sample_length, csi_features, mems_features)

    # 4. TensorRT (FP16 + FP32)
    if not args.skip_trt:
        export_tensorrt(onnx_path, output_dir, sample_length, csi_features, mems_features)
    else:
        print("[EXPORT] Skipping TensorRT export (--skip-trt)")

    # 5. Copy OC-SVM models if they exist alongside checkpoint
    ckpt_dir = Path(args.checkpoint).parent
    ocsvm_dir = ckpt_dir / "ocsvm_models"
    if ocsvm_dir.exists():
        dest_ocsvm = output_dir / "ocsvm_models"
        if dest_ocsvm.exists():
            shutil.rmtree(dest_ocsvm)
        shutil.copytree(ocsvm_dir, dest_ocsvm)
        print(f"[EXPORT] OC-SVM models: {dest_ocsvm}")

    print("[EXPORT] Done!")


if __name__ == "__main__":
    main()
