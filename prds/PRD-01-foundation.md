# PRD-01: Foundation

## Objective
Set up the project structure, configuration system, data loading pipeline, and CSI parser.

## Deliverables
1. `pyproject.toml` with all dependencies and hatchling backend
2. `configs/paper.toml` with paper-exact hyperparameters
3. `configs/debug.toml` for quick smoke tests
4. `src/securelink/__init__.py` with package metadata
5. `src/securelink/dataset.py` with:
   - CSI binary parser (PicoScenes .csi format)
   - Telemetry CSV loader
   - Phase error extraction from raw CSI
   - Outlier filtering (phase gradient variance threshold eta=4)
   - MEMS 8-field selection (pitch, roll, yaw, tof, baro, agx, agy, agz)
   - Linear interpolation alignment of CSI to telemetry timestamps
   - Sample construction (M=6 frames per sample)
   - Dataset A (closed-world) and Dataset B (open-world) splitting
   - PyTorch Dataset class with proper train/val/test splits
6. `src/securelink/utils.py` with config loading, seeding, device selection
7. `tests/test_dataset.py` with smoke tests

## Acceptance Criteria
- [ ] `uv sync` succeeds without errors
- [ ] Dataset loads and returns correct tensor shapes
- [ ] CSI phase errors extracted from .csi files match expected dimensions (K subcarriers)
- [ ] Telemetry loads 8 fields per frame
- [ ] Alignment produces M matched CSI+telemetry frames per sample
- [ ] Train/val/test splits are reproducible with seed=42

## Technical Notes
- CSI .csi files are PicoScenes binary format -- need custom parser
- 52 subcarriers total, mirror subcarriers used for lambda/z estimation
- Phase error dimension K depends on subcarrier count after processing
- Telemetry at ~10Hz, CSI much sparser -- interpolation upsamples CSI
