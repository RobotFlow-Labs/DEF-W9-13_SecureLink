# PRD-07: Integration

## Objective
Docker serving infrastructure, health checks, and ANIMA module registration.

## Deliverables
1. `Dockerfile.serve` -- 3-layer build from anima-serve:jazzy
2. `docker-compose.serve.yml` -- profiles: serve, ros2, api, test
3. `.env.serve` -- module environment variables
4. `src/securelink/serve.py` -- AnimaNode subclass
   - `setup_inference()`: Load DNN + OC-SVM models
   - `process()`: Accept CSI+MEMS data, return UAV auth result
   - `get_status()`: Report model loaded state
5. `anima_module.yaml` -- module manifest
6. Health endpoints:
   - GET /health -- status, uptime, GPU VRAM
   - GET /ready -- weights loaded check
   - GET /info -- module info
   - POST /predict -- authenticate UAV from CSI+MEMS input

## API Specification
```
POST /predict
{
  "csi_data": [[...], ...],      // M x K CSI phase errors
  "mems_data": [[...], ...],     // M x 8 telemetry values
  "claimed_uav_id": 5            // UAV ID to verify
}
Response:
{
  "authenticated": true/false,
  "confidence": 0.95,
  "matched_uav_id": 5,
  "all_scores": {0: 0.1, ..., 21: 0.05}
}
```

## Acceptance Criteria
- [ ] Docker image builds successfully
- [ ] Health endpoint returns 200
- [ ] Predict endpoint returns correct JSON schema
- [ ] anima_module.yaml passes validation
