# SecureLink -- Asset Inventory

## Datasets

### SecureLink Dataset (INCLUDED)
- **Location**: `/mnt/forge-data/modules/05_wave9/13_SecureLink/repositories/SecureLink_data/`
- **Size**: 208MB
- **Format**:
  - `CSI_{00-21}/` -- Binary .csi files (PicoScenes format) per UAV, 25 trials each
  - `sensors_{00-21}/` -- CSV telemetry files per UAV, 25 trials each
  - Telemetry CSV columns: TS, pitch, roll, yaw, vgx, vgy, vgz, templ, temph, tof, h, bat, baro, time, agx, agy, agz
- **UAV IDs**:
  - 00-19: DJI Tello drones
  - 20: DJI Phantom 4 Pro with ESP32 SoC
  - 21: DJI Phantom 4 Pro with ESP32-S3 SoC
- **Trial labels**:
  - 1-5: Rooftop (dynamic flight)
  - 6-10: Playground (dynamic flight)
  - 11-15: Corridor (dynamic flight)
  - 16-20: Office (dynamic flight)
  - 21-25: Office (static, motors off)
- **Source**: https://github.com/PhyGroup/SecureLink
- **Status**: AVAILABLE -- already cloned

### Symlink for Shared Access
```bash
# Optional: symlink to shared datasets dir
ln -sf /mnt/forge-data/modules/05_wave9/13_SecureLink/repositories/SecureLink_data \
       /mnt/forge-data/datasets/securelink
```

## Pretrained Models
- **None required** -- SecureLink trains from scratch
- No foundation models, no pretrained backbones

## External Dependencies
- **PicoScenes**: CSI collection tool (not needed for training, only for data collection)
  - We need a CSI parser to read .csi binary files
  - The paper uses PicoScenes format -- we implement a custom parser

## Shared Infrastructure
- **Path**: `/mnt/forge-data/shared_infra/`
- **Relevant caches**: None (SecureLink uses custom CSI+MEMS data, not vision features)
- **CUDA extensions**: None needed (pure PyTorch + sklearn)

## Output Artifacts
All training outputs go to `/mnt/artifacts-datai/`:
```
/mnt/artifacts-datai/
├── checkpoints/securelink/     -- model checkpoints
├── models/securelink/          -- final trained models
├── logs/securelink/            -- training logs
├── exports/securelink/         -- ONNX, TorchScript exports
├── reports/securelink/         -- eval reports
└── tensorboard/securelink/     -- TensorBoard events
```

## Disk Budget
| Asset | Size | Status |
|-------|------|--------|
| SecureLink dataset | 208MB | AVAILABLE |
| Model checkpoints | ~50MB | Will be generated |
| OC-SVM models | ~10MB | Will be generated |
| ONNX export | ~10MB | Will be generated |
| **Total** | **~280MB** | |
