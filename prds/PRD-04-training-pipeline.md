# PRD-04: Training Pipeline

## Objective
Implement the full training loop with checkpointing, logging, OC-SVM fitting, and resume.

## Deliverables
1. `src/securelink/train.py` with:
   - Config-driven training loop (TOML + Pydantic)
   - Adam optimizer, lr=0.001
   - Cosine annealing LR scheduler with 5% warmup
   - Mixed precision (bf16 on CUDA)
   - Gradient clipping (max_norm=1.0)
   - Step-based checkpointing (every 500 steps, keep top 2)
   - Early stopping (patience=10 epochs)
   - NaN detection
   - TensorBoard logging
   - Resume from checkpoint support
   - Seed reproducibility (42)
2. `scripts/train.py` entry point
3. OC-SVM fitting after DNN training:
   - Extract embeddings for all registered UAVs
   - Fit one OC-SVM per UAV using scikit-learn
   - Save OC-SVM models alongside DNN checkpoint

## Training Flow
1. Load config from TOML
2. Build datasets (Dataset A or B)
3. Build model, optimizer, scheduler
4. Training loop:
   - Forward: (csi, mems) -> embeddings
   - Multi-similarity loss
   - Backward + clip gradients + step
   - Validate every epoch
   - Checkpoint best models
5. After DNN training:
   - Extract all training embeddings
   - Fit OC-SVM per UAV
   - Save complete model (DNN + OC-SVMs)

## Acceptance Criteria
- [ ] Smoke test: 2 epochs on subset, loss decreases
- [ ] Checkpoint save/load/resume cycle works
- [ ] OC-SVM fitting produces per-UAV classifiers
- [ ] Logs written to /mnt/artifacts-datai/logs/securelink/
- [ ] TensorBoard events written
