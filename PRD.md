# SecureLink -- Product Requirements Document

## Module Identity
- **Name**: SecureLink
- **Package**: securelink
- **Paper**: Securing UAV Communications by Fusing Cross-Layer Fingerprints (2511.05796)
- **Domain**: Defense -- UAV Authentication / Anti-Impersonation
- **Git prefix**: [SECURELINK]

## Problem Statement
UAV communications over wireless links are vulnerable to impersonation attacks where
adversaries use stolen digital certificates. Existing fingerprint-based authentication
relies on single-modality data from one network layer, producing unreliable results
in dynamic open-world environments. SecureLink fuses physical-layer RF fingerprints
(CSI phase errors) with application-layer MEMS fingerprints (accelerometer, barometer,
ToF) via attention-based multimodal fusion and one-class SVM for open-world authentication.

## Target Metrics (from paper)
| Metric | Closed-World | Open-World |
|--------|-------------|------------|
| Accuracy | 98.61% | 97.54% |
| TNR | 99.04% | 96.95% |
| Recall | >97% | >96% |
| Precision | >98% | >97% |
| Runtime | <15ms | <15ms |

## Build Plan

| PRD | Title | Status | Description |
|-----|-------|--------|-------------|
| PRD-01 | Foundation | NOT STARTED | Project structure, configs, data loaders, CSI parser |
| PRD-02 | Core Model | NOT STARTED | Two-branch CNN+BiLSTM, attention fusion, embedding |
| PRD-03 | Loss Functions | NOT STARTED | Multi-similarity loss implementation |
| PRD-04 | Training Pipeline | NOT STARTED | Training loop, checkpointing, OC-SVM fitting |
| PRD-05 | Evaluation | NOT STARTED | Metrics, closed/open-world eval, confusion matrix |
| PRD-06 | Export Pipeline | NOT STARTED | pth, safetensors, ONNX, TensorRT export |
| PRD-07 | Integration | NOT STARTED | Docker, serve.py, health checks, anima_module.yaml |

## Architecture Overview
```
Raw CSI (.csi) ──> Phase Error Extraction ──> Outlier Filter (eta=4)
                                                    │
Raw Telemetry (.csv) ──> 8-field Selection ──> Outlier Cleaning
                                                    │
              ┌─────────── Linear Interpolation Alignment ───────────┐
              │                                                       │
         CSI Sequence (M x K)                              Telemetry Sequence (M x 8)
              │                                                       │
         1D-CNN (64,64)                                      1D-CNN (64,64)
              │                                                       │
         BiLSTM (128)                                        BiLSTM (128)
              │                                                       │
         x^m (128-dim)                                       y^m (128-dim)
              │                                                       │
              └──────────── Concatenate (256-dim) ────────────────────┘
                                      │
                        Multi-Head Attention x2 (4 heads)
                                      │
                              Flatten -> D
                                      │
                            FC Embedding (256-dim) -> g
                                      │
                    ┌─────────────────┴─────────────────┐
              Multi-Similarity Loss              OC-SVM (per UAV)
              (Training phase)                   (Inference phase)
```

## Dependencies
- torch >= 2.0 (cu128)
- numpy
- scipy
- scikit-learn (OC-SVM)
- pandas (CSV loading)
- tomli / pydantic (config)
- tensorboard
- safetensors
- onnx, onnxruntime
- ruff (linting)
- pytest (testing)
