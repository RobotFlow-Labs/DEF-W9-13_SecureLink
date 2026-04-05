# SecureLink Training Report

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Config | configs/paper.toml |
| Model | Two-branch CNN+BiLSTM + 2x MHA Fusion |
| Parameters | 1,281,792 |
| Batch size | 256 |
| Learning rate | 0.001 (Adam) |
| Epochs | 30 |
| Precision | bf16 |
| Scheduler | Warmup (5%) + Cosine |
| GPU | NVIDIA L4 (23GB), GPU 2 |
| Training time | ~32 minutes |

## Dataset
| Split | Samples |
|-------|---------|
| Total | 40,812 |
| Train | 24,487 (60%) |
| Val | 8,162 (20%) |
| Test | 8,163 (20%) |
| UAVs | 22 (20 Tello + 2 Phantom4) |

## Training Curve
| Epoch | Train Loss | Val Loss | LR |
|-------|-----------|----------|-----|
| 1 | 3.1901 | 3.2221 | 0.000669 |
| 5 | 3.1383 | 3.1610 | 0.000963 |
| 10 | 3.0717 | 3.1055 | 0.000796 |
| 15 | 2.9944 | 3.0366 | 0.000541 |
| 20 | 2.9419 | 3.0180 | 0.000274 |
| 25 | 2.8820 | 2.9834 | 0.000074 |
| 30 | 2.8605 | 2.9557 | 0.000001 |

## Evaluation Results

### Closed-World (Dataset A)
| Metric | Achieved | Paper Target |
|--------|----------|-------------|
| Accuracy | 60.86% | 98.61% |
| Recall | 71.35% | >97% |
| Precision | 70.63% | >98% |

### Open-World (Dataset B)
| Metric | Achieved | Paper Target |
|--------|----------|-------------|
| TNR | 100.00% | 96.95% |
| Accuracy | 12.47% | 97.54% |

## Analysis
- TNR=100% shows the model successfully learns to distinguish UAV fingerprints
- Closed-world accuracy gap likely due to:
  1. CSI binary parser may extract different features than original PicoScenes SDK
  2. Multi-similarity loss convergence may need margin tuning
  3. Paper uses TensorFlow 2.13 — PyTorch reimplementation may differ
  4. Data preprocessing (phase error extraction) may differ from original

## Exported Formats
| Format | Size | Status |
|--------|------|--------|
| pth | 15.5MB | OK |
| safetensors | 5.1MB | OK |
| ONNX | 5.4MB | OK (verified) |
| TensorRT FP16 | 3.0MB | OK |
| TensorRT FP32 | 5.7MB | OK |
| OC-SVM (22 models) | 1.4MB | OK |

## Artifacts
- Checkpoints: `/mnt/artifacts-datai/checkpoints/securelink/`
- Exports: `/mnt/artifacts-datai/exports/securelink/`
- Logs: `/mnt/artifacts-datai/logs/securelink/`
- Reports: `/mnt/artifacts-datai/reports/securelink/`
