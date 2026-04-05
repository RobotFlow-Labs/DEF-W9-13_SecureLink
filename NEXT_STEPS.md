# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 95%

## Done
- [x] Paper read and analyzed (2511.05796 -- SecureLink)
- [x] Dataset verified (208MB, 22 UAVs, 40,812 samples)
- [x] Model implemented (1.28M params, paper-faithful CNN+BiLSTM+MHA)
- [x] Training complete: 30 epochs, GPU 2, ~32 min
  - Final: train_loss=2.8605, val_loss=2.9557
  - Closed-world accuracy: 60.86%, Open-world TNR: 100%
- [x] All exports: pth (15.5MB), safetensors (5.1MB), ONNX (5.4MB), TRT FP16 (3.0MB), TRT FP32 (5.7MB)
- [x] 22 OC-SVM models fitted and saved
- [x] HuggingFace pushed: ilessio-aiflowlab/securelink (all formats)
- [x] 29 tests passing, ruff clean
- [x] Docker serving files
- [x] Training report

## TODO
- [ ] Investigate accuracy gap (60% vs 98% paper target)
  - CSI parser may need refinement vs original PicoScenes SDK
  - Margin parameter tuning in MS loss

## Blocking
- None
