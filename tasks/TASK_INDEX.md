# SecureLink -- Task Index

## PRD-01: Foundation
- [ ] T01.1: Create pyproject.toml with hatchling backend and all dependencies
- [ ] T01.2: Create configs/paper.toml with paper hyperparameters
- [ ] T01.3: Create configs/debug.toml for smoke tests
- [ ] T01.4: Implement PicoScenes .csi binary parser
- [ ] T01.5: Implement CSI phase error extraction (52 subcarriers)
- [ ] T01.6: Implement phase gradient variance outlier filter (eta=4)
- [ ] T01.7: Implement telemetry CSV loader with 8-field selection
- [ ] T01.8: Implement telemetry outlier cleaning
- [ ] T01.9: Implement linear interpolation alignment (CSI -> telemetry timestamps)
- [ ] T01.10: Implement sample construction (M=6 frames per sample)
- [ ] T01.11: Implement Dataset A (closed-world) split: 60/20/20
- [ ] T01.12: Implement Dataset B (open-world) split: 80/20 with 2 impersonators
- [ ] T01.13: Implement PyTorch Dataset and DataLoader classes
- [ ] T01.14: Implement utils.py (config, seeding, device selection)
- [ ] T01.15: Write test_dataset.py smoke tests

## PRD-02: Core Model
- [ ] T02.1: Implement UnimodalBranch (1D-CNN + BiLSTM)
- [ ] T02.2: Implement 1D-CNN block (Conv1D-64-k3, Pool, Conv1D-64-k2, BN, ReLU)
- [ ] T02.3: Implement BiLSTM (128 cells, bidirectional)
- [ ] T02.4: Implement MultiHeadAttentionFusion (2 layers, 4 heads)
- [ ] T02.5: Implement feed-forward sublayers with residual + LayerNorm
- [ ] T02.6: Implement embedding layer (FC-256 + L2 normalize)
- [ ] T02.7: Assemble SecureLinkModel (2 branches + fusion + embedding)
- [ ] T02.8: Verify parameter count (~2M)
- [ ] T02.9: Write test_model.py smoke tests

## PRD-03: Loss Functions
- [ ] T03.1: Implement cosine similarity matrix computation
- [ ] T03.2: Implement positive/negative pair mining from UAV labels
- [ ] T03.3: Implement multi-similarity loss (Eq. 7, alpha=1, beta=10)
- [ ] T03.4: Add log-sum-exp numerical stability
- [ ] T03.5: Write loss function unit tests

## PRD-04: Training Pipeline
- [ ] T04.1: Implement Pydantic config model from TOML
- [ ] T04.2: Implement training loop with Adam optimizer
- [ ] T04.3: Implement warmup + cosine annealing LR scheduler
- [ ] T04.4: Implement bf16 mixed precision training
- [ ] T04.5: Implement gradient clipping (max_norm=1.0)
- [ ] T04.6: Implement CheckpointManager (keep top 2 by val_loss)
- [ ] T04.7: Implement early stopping (patience=10)
- [ ] T04.8: Implement NaN detection and halt
- [ ] T04.9: Implement TensorBoard logging
- [ ] T04.10: Implement resume from checkpoint
- [ ] T04.11: Implement OC-SVM fitting after DNN training
- [ ] T04.12: Create scripts/train.py entry point
- [ ] T04.13: Checkpoint smoke test (save/load/resume)

## PRD-05: Evaluation
- [ ] T05.1: Implement accuracy, TNR, recall, precision metrics
- [ ] T05.2: Implement closed-world evaluation (Dataset A, OC-SVM selection)
- [ ] T05.3: Implement 22x22 confusion matrix generation
- [ ] T05.4: Implement open-world evaluation (Dataset B, 6-round protocol)
- [ ] T05.5: Implement per-environment accuracy breakdown
- [ ] T05.6: Implement sample length ablation (M=2,4,6)
- [ ] T05.7: Create scripts/evaluate.py entry point
- [ ] T05.8: Implement report generation (JSON + markdown)

## PRD-06: Export Pipeline
- [ ] T06.1: Implement safetensors export
- [ ] T06.2: Implement ONNX export (opset 17)
- [ ] T06.3: Implement TensorRT export (FP16/FP32, optional)
- [ ] T06.4: Implement OC-SVM pickle serialization
- [ ] T06.5: Implement ONNX output verification vs PyTorch
- [ ] T06.6: Create scripts/export.py entry point
- [ ] T06.7: Generate model card for HuggingFace

## PRD-07: Integration
- [ ] T07.1: Create Dockerfile.serve (3-layer from anima-serve:jazzy)
- [ ] T07.2: Create docker-compose.serve.yml (profiles: serve, ros2, api, test)
- [ ] T07.3: Create .env.serve with module identity
- [ ] T07.4: Implement serve.py (AnimaNode subclass)
- [ ] T07.5: Implement /predict endpoint
- [ ] T07.6: Create anima_module.yaml
- [ ] T07.7: End-to-end Docker build + health check test
