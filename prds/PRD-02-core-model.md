# PRD-02: Core Model

## Objective
Implement the SecureLink two-branch neural network with attention-based multimodal fusion.

## Deliverables
1. `src/securelink/model.py` with:
   - `UnimodalBranch`: 1D-CNN (2 conv layers, 64 filters each) + BiLSTM (128 cells)
   - `MultiHeadAttentionFusion`: 2 self-attention layers with 4 heads each
   - `SecureLinkModel`: Full model combining two branches, fusion, and embedding layer
   - Forward pass: (csi_batch, mems_batch) -> 256-dim embedding vectors

## Architecture Details

### UnimodalBranch (shared architecture for both CSI and MEMS)
```
Input: (batch, seq_len, features)
  -> Conv1D(in_feat, 64, kernel=3, padding=1) + BN + ReLU
  -> MaxPool1D(2)
  -> Conv1D(64, 64, kernel=2, padding=0) + BN + ReLU
  -> BiLSTM(input_size=64, hidden_size=64, bidirectional=True)
  -> Output: (batch, 128)  # last hidden state concatenated
```

### MultiHeadAttentionFusion
```
Input: concatenated features (batch, seq, 256)
  -> MultiHeadAttention(embed_dim=256, num_heads=4) + residual + LayerNorm
  -> FeedForward(256 -> 512 -> 256) + residual + LayerNorm
  -> MultiHeadAttention(embed_dim=256, num_heads=4) + residual + LayerNorm
  -> FeedForward(256 -> 512 -> 256) + residual + LayerNorm
  -> Flatten -> multimodal vector D
```

### Embedding Layer
```
D -> Linear(D_dim, 256) -> L2 normalize -> g (256-dim embedding)
```

## Acceptance Criteria
- [ ] Model forward pass produces (batch, 256) output
- [ ] Parameter count ~2M (verify)
- [ ] Both branches can handle variable input feature dimensions
- [ ] Gradient flow verified through all components
- [ ] Smoke test with random input passes
