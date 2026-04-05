# PRD-05: Evaluation

## Objective
Implement evaluation metrics, closed-world and open-world evaluation protocols.

## Deliverables
1. `src/securelink/evaluate.py` with:
   - Accuracy, TNR, Recall, Precision computation
   - Closed-world evaluation (Dataset A):
     - Per-UAV OC-SVM prediction
     - Select highest-matching OC-SVM score
     - 22-class confusion matrix
   - Open-world evaluation (Dataset B):
     - 20 registered + 2 impersonating UAVs
     - Detect impersonators via OC-SVM rejection
     - 6-round evaluation protocol
   - Confusion matrix visualization
   - Per-environment accuracy breakdown
   - Sample length ablation (M=2,4,6)
2. `scripts/evaluate.py` entry point
3. Report generation (metrics JSON + summary)

## Target Metrics
| Setting | Accuracy | TNR |
|---------|----------|-----|
| Closed-world | 98.61% | 99.04% |
| Open-world (avg 6 rounds) | 97.54% | 96.95% |

## Acceptance Criteria
- [ ] All 4 metrics computed correctly
- [ ] Confusion matrix matches paper format (22x22)
- [ ] Open-world 6-round protocol implemented
- [ ] Reports saved to /mnt/artifacts-datai/reports/securelink/
