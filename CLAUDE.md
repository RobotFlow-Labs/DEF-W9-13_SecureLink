# SecureLink -- Cross-Layer UAV Authentication via RF and MEMS Fingerprint Fusion

## Paper
- **Title**: Securing UAV Communications by Fusing Cross-Layer Fingerprints
- **Authors**: Yong Huang, Ruihao Li, Mingyang Chen, Feiyang Zhao, Dalong Zhang, Wanqing Tu
- **Venue**: IEEE Internet of Things Journal, 2025
- **arXiv**: 2511.05796
- **GitHub**: https://github.com/PhyGroup/SecureLink

## Summary
SecureLink is a UAV authentication system that fuses physical-layer RF fingerprints
(Wi-Fi CSI phase errors) with application-layer MEMS fingerprints (accelerometer,
gyroscope, barometer, ToF) for cross-layer multimodal authentication. It addresses
UAV impersonation attacks where adversaries steal digital certificates to pose as
legitimate drones. The system:
1. Extracts CSI phase errors as RF fingerprints from the physical layer
2. Extracts 8 MEMS sensor fields (pitch, roll, yaw, tof, baro, agx, agy, agz) from telemetry
3. Aligns the two modalities via linear interpolation (CSI is sparser than telemetry)
4. Feeds aligned sequences through a two-branch 1D-CNN + BiLSTM network
5. Fuses features via two multi-head attention layers (4 heads each)
6. Trains with multi-similarity loss for UAV registration
7. Uses per-UAV one-class SVMs (OC-SVM) for open-world authentication

## Architecture Details

### Data Preprocessing
- **RF fingerprints**: Extract CSI phase errors e = phi - psi - 2*pi*lambda*i - z
  from 52 subcarriers. Filter outlier frames by phase gradient variance threshold eta=4.
- **MEMS fingerprints**: 8 fields from telemetry: pitch, roll, yaw, tof, baro, agx, agy, agz.
  Clean frames with outlier values.
- **Alignment**: Linear interpolation to upsample CSI frames to match telemetry frame count.
  CSI frames L << M telemetry frames. Result: M aligned CSI+telemetry frame pairs.
- **Sample length**: M=6 data frames per fingerprint sample (optimal trade-off).

### Unimodal Feature Extraction (per branch)
Each branch (CSI and MEMS) has identical architecture:
1. **1D-CNN**:
   - Conv1D(in_channels, 64, kernel_size=3) + BN + ReLU
   - MaxPool1D(2)
   - Conv1D(64, 64, kernel_size=2) + BN + ReLU
2. **BiLSTM**: 128 cells (64 per direction), processes CNN output

Output: 128-dim unimodal feature vector per modality per sample.

### Multimodal Feature Fusion
1. Concatenate CSI and MEMS unimodal features -> feature map U (256-dim)
2. Two sequential multi-head self-attention layers (4 heads each)
   - Q, K, V projections, scaled dot-product attention
   - Feed-forward after each attention layer
3. Flatten to 1D multimodal feature vector D

### UAV Identification
1. **Embedding layer**: FC(256) compresses multimodal vector to 256-dim embedding g
2. **Multi-similarity loss**: Trains the embedding space
   - alpha=1, beta=10, mu (margin) tuned per dataset
3. **OC-SVM**: Per-UAV one-class SVM with Gaussian kernel for open-world auth
   - tau in [0,1] controls regularization vs margin trade-off

## Hyperparameters (from paper)
| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Batch size | 256 |
| Epochs | 30 |
| Optimizer | Adam |
| Sample length M | 6 frames |
| CNN kernel sizes | 3, 2 |
| CNN filters | 64, 64 |
| BiLSTM cells | 128 (64 per direction) |
| Attention heads | 4 |
| Attention layers | 2 |
| Embedding dim | 256 |
| MS loss alpha | 1 |
| MS loss beta | 10 |
| Phase variance threshold eta | 4 |
| CSI subcarriers | 52 (26 used after mirror removal) |
| MEMS fields | 8 (pitch, roll, yaw, tof, baro, agx, agy, agz) |
| OC-SVM kernel | RBF (Gaussian) |

## Dataset
- **Source**: https://github.com/PhyGroup/SecureLink (208MB included in repositories/)
- **UAVs**: 22 total (20 DJI Tello + 1 Phantom4Pro-ESP32 + 1 Phantom4Pro-ESP32S3)
- **Environments**: 4 (rooftop, playground, corridor, office) + 1 static (office)
- **Trials per env**: 5-6 dynamic + 5 static per UAV
- **Total samples**: ~400K aligned data frames, ~20K per UAV
- **Dataset A**: Closed-world (all 22 UAVs, 60/20/20 train/val/test split)
- **Dataset B**: Open-world (20 registered + 2 impersonating, 80/20 split)

## Model Requirements
- Framework: PyTorch (original used TensorFlow 2.13.1, we reimplement in PyTorch)
- No pretrained models needed (trains from scratch)
- scikit-learn for OC-SVM
- PicoScenes CSI parser (or custom binary parser for .csi files)

## Training Infrastructure
- Single GPU sufficient (model is small, ~2M parameters)
- VRAM: <2GB estimated
- Training time: ~5-10 min on L4

## Git Commit Prefix
[SECURELINK]
