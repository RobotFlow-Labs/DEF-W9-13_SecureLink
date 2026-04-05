# NEXT_STEPS.md
> Last updated: 2026-04-05
> MVP Readiness: 15%

## Done
- [x] Paper read and analyzed (2511.05796 -- SecureLink)
- [x] Dataset inspected (208MB, 22 UAVs, CSI + telemetry)
- [x] Project scaffolding created (CLAUDE.md, ASSETS.md, PRD.md, 7 PRDs, tasks)
- [x] Python package structure created (src/securelink/)
- [x] Model architecture implemented (model.py -- CNN+BiLSTM+Attention+Embedding)
- [x] Loss function implemented (losses.py -- MultiSimilarityLoss)
- [x] Dataset loader implemented (dataset.py -- CSI parser, telemetry loader, alignment)
- [x] Training pipeline implemented (train.py)
- [x] Evaluation pipeline implemented (evaluate.py)
- [x] Utils implemented (utils.py -- config, seeding, device)
- [x] Config files created (paper.toml, debug.toml)
- [x] Entry scripts created (scripts/train.py, scripts/evaluate.py)
- [x] Tests created (test_model.py, test_dataset.py)
- [x] Docker serving files created (Dockerfile.serve, docker-compose.serve.yml)
- [x] anima_module.yaml created

## In Progress
- [ ] None

## TODO
- [ ] Create and activate .venv, install dependencies
- [ ] Run smoke tests to verify dataset loading
- [ ] Debug CSI binary parser against actual .csi files
- [ ] Run training on GPU (PRD-04)
- [ ] Run evaluation (PRD-05)
- [ ] Export models (PRD-06)
- [ ] Build and test Docker image (PRD-07)

## Blocking
- CSI .csi file format is PicoScenes binary -- parser implemented based on header
  inspection but needs verification against actual data
- Need GPU allocation for training

## Downloads Needed
- None -- dataset already available in repositories/SecureLink_data/
