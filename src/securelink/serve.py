"""SecureLink serving: AnimaNode subclass for Docker deployment.

Exposes:
  GET /health -- system health
  GET /ready  -- model readiness
  GET /info   -- module info
  POST /predict -- UAV authentication
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import torch

from securelink.model import SecureLinkModel, build_model
from securelink.utils import ModelConfig, get_device


class SecureLinkServer:
    """SecureLink inference server.

    Loads the DNN model and OC-SVM classifiers, provides authentication API.
    """

    def __init__(
        self,
        weights_dir: str | Path = "/data/weights",
        device: str = "auto",
    ):
        self.weights_dir = Path(weights_dir)
        self.device = get_device(device)
        self.model: SecureLinkModel | None = None
        self.ocsvms: dict[int, Any] = {}
        self.start_time = time.time()
        self._ready = False

    def setup_inference(self) -> None:
        """Load model weights and OC-SVM classifiers."""
        # Load DNN
        ckpt_path = self.weights_dir / "best.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            config_dict = ckpt.get("config", {})
            model_config = ModelConfig(**config_dict.get("model", {}))

            self.model = build_model(model_config)
            self.model.load_state_dict(ckpt["model"])
            self.model.to(self.device)
            self.model.eval()
            print("[SERVE] DNN model loaded")
        else:
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

        # Load OC-SVMs
        ocsvm_dir = self.weights_dir / "ocsvm_models"
        if ocsvm_dir.exists():
            for pkl_file in sorted(ocsvm_dir.glob("ocsvm_uav_*.pkl")):
                uav_id = int(pkl_file.stem.split("_")[-1])
                with open(pkl_file, "rb") as f:
                    self.ocsvms[uav_id] = pickle.load(f)
            print(f"[SERVE] Loaded {len(self.ocsvms)} OC-SVM models")

        self._ready = True

    def process(
        self,
        csi_data: list[list[float]],
        mems_data: list[list[float]],
        claimed_uav_id: int | None = None,
    ) -> dict[str, Any]:
        """Run authentication inference.

        Args:
            csi_data: (M, K) CSI phase error values
            mems_data: (M, 8) MEMS telemetry values
            claimed_uav_id: Optional UAV ID to verify

        Returns:
            Authentication result dict
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call setup_inference() first.")

        # Convert to tensors
        csi = torch.tensor([csi_data], dtype=torch.float32, device=self.device)
        mems = torch.tensor([mems_data], dtype=torch.float32, device=self.device)

        # Get embedding
        with torch.no_grad():
            embedding = self.model(csi, mems).cpu().numpy()[0]

        # Run OC-SVMs
        scores = {}
        for uav_id, svm in self.ocsvms.items():
            score = svm.decision_function(embedding.reshape(1, -1))[0]
            scores[uav_id] = float(score)

        # Find best match
        if scores:
            best_id = max(scores, key=scores.get)
            best_score = scores[best_id]
            authenticated = best_score > 0
        else:
            best_id = -1
            best_score = -1.0
            authenticated = False

        # If claimed_uav_id specified, check that specific OC-SVM
        if claimed_uav_id is not None and claimed_uav_id in self.ocsvms:
            claimed_score = scores.get(claimed_uav_id, -1.0)
            authenticated = claimed_score > 0 and best_id == claimed_uav_id

        return {
            "authenticated": authenticated,
            "confidence": float(best_score),
            "matched_uav_id": int(best_id),
            "claimed_uav_id": claimed_uav_id,
            "all_scores": scores,
        }

    def get_health(self) -> dict[str, Any]:
        """Return health status."""
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_vram_mb": torch.cuda.get_device_properties(0).total_mem // (1024 * 1024),
            }
        return {
            "status": "healthy",
            "module": "securelink",
            "uptime_s": time.time() - self.start_time,
            **gpu_info,
        }

    def get_ready(self) -> dict[str, Any]:
        """Return readiness status."""
        return {
            "ready": self._ready,
            "module": "securelink",
            "version": "0.1.0",
            "weights_loaded": self.model is not None,
            "ocsvms_loaded": len(self.ocsvms),
        }

    def get_info(self) -> dict[str, Any]:
        """Return module info."""
        return {
            "name": "securelink",
            "version": "0.1.0",
            "description": "Cross-layer UAV authentication via RF+MEMS fingerprint fusion",
            "paper": "2511.05796",
            "capabilities": [
                "uav-authentication",
                "impersonation-detection",
                "rf-fingerprinting",
                "multimodal-fusion",
            ],
            "inputs": {
                "csi_data": "M x K float array (CSI phase errors)",
                "mems_data": "M x 8 float array (telemetry: pitch,roll,yaw,tof,baro,agx,agy,agz)",
            },
        }
