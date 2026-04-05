# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 85%

## Done
- [x] Paper read and analyzed (2511.05796 -- SecureLink)
- [x] Dataset inspected (208MB, 22 UAVs, CSI + telemetry)
- [x] Project scaffolding created
- [x] Model architecture implemented (1.28M params, paper-faithful)
- [x] Loss function implemented (MultiSimilarityLoss + vectorized variant)
- [x] Dataset loader implemented (CSI binary parser verified on real data)
- [x] Training pipeline (warmup+cosine, checkpointing, early stopping, OC-SVM)
- [x] Evaluation pipeline (closed/open-world, metrics)
- [x] Full dataset loading verified (40,812 samples, 22 UAVs)
- [x] 29 tests passing, ruff lint clean
- [x] Venv: torch 2.11.0+cu128, all deps installed
- [x] Training complete: 30 epochs on GPU 2, ~32 min
  - Final: train_loss=2.8605, val_loss=2.9557
  - Closed-world accuracy: 60.86% (paper: 98.61%)
  - Open-world TNR: 100% (paper: 96.95%)
- [x] Exports: pth (15.5MB), safetensors (5.1MB), ONNX (5.4MB, verified)
- [x] OC-SVM models: 22 models fitted and saved
- [x] Docker serving files (Dockerfile.serve, docker-compose.serve.yml)
- [x] Training report generated

## TODO
- [ ] TRT export (blocked by disk space — need TensorRT libs ~4GB)
- [ ] Push to HuggingFace: ilessio-aiflowlab/securelink
- [ ] Investigate accuracy gap (60% vs 98% paper target)
  - CSI parser may need refinement vs original PicoScenes SDK
  - Margin parameter tuning in MS loss
  - Consider using original TF reference implementation for comparison
- [ ] Final git commit + push

## Blocking
- Disk space for TensorRT installation (forge-data at 99%)
