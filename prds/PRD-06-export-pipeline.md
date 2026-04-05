# PRD-06: Export Pipeline

## Objective
Export trained models to multiple formats for deployment.

## Deliverables
1. Export pipeline in `src/securelink/export.py`:
   - PyTorch .pth (already saved during training)
   - safetensors format
   - ONNX export (opset 17)
   - TensorRT FP16/FP32 (if tensorrt available)
2. `scripts/export.py` entry point
3. Model card generation for HuggingFace

## Export Details
- DNN model: standard PyTorch -> ONNX conversion
  - Input: (batch, M, K_csi) and (batch, M, 8)
  - Output: (batch, 256) embedding
- OC-SVM models: pickle serialization (scikit-learn)
- Combined package: DNN weights + OC-SVM models + config

## Output Location
```
/mnt/artifacts-datai/exports/securelink/
├── securelink_dnn.pth
├── securelink_dnn.safetensors
├── securelink_dnn.onnx
├── securelink_dnn_fp16.trt  (if TensorRT available)
├── ocsvm_models/
│   ├── ocsvm_uav_00.pkl
│   ├── ...
│   └── ocsvm_uav_21.pkl
└── model_card.md
```

## Acceptance Criteria
- [ ] ONNX model produces same output as PyTorch (rtol=1e-4)
- [ ] safetensors loads correctly
- [ ] OC-SVM models load and predict correctly
- [ ] All exports saved to /mnt/artifacts-datai/exports/securelink/
