# PRD-03: Loss Functions

## Objective
Implement the multi-similarity loss from the paper for metric learning.

## Deliverables
1. `src/securelink/losses.py` with:
   - `MultiSimilarityLoss`: Paper equation (7)
     - Cosine similarity between embedding pairs
     - Positive pairs: same UAV ID
     - Negative pairs: different UAV IDs
     - Parameters: alpha=1, beta=10, mu (margin)
   - Mining of positive and negative pairs per anchor
   - Numerical stability (log-sum-exp trick)

## Multi-Similarity Loss (Equation 7)
```
L_MS = (1/A) * sum_i [
    (1/alpha) * log(1 + sum_{j in P_i} exp(-alpha * (C_ij - mu)))
  + (1/beta)  * log(1 + sum_{r in N_i} exp( beta * (C_ir - mu)))
]
```
Where:
- A = number of anchor samples in batch
- P_i = positive set for anchor i (same UAV)
- N_i = negative set for anchor i (different UAVs)
- C_ij = cosine similarity between anchor i and sample j
- mu = margin
- alpha = 1, beta = 10 (from paper)

## Acceptance Criteria
- [ ] Loss computes correct gradients
- [ ] Handles edge cases (no positives, no negatives)
- [ ] Numerically stable with large batches
- [ ] Verified: positive pairs get higher similarity, negatives get lower
