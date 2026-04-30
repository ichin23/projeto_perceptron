# Requirements: Perceptron Oil Purity Classifier

**Defined:** 2026-04-30
**Core Value:** Accurately classify oil purity patterns using a manual Perceptron implementation without high-level libraries like scikit-learn.

## v1 Requirements

### Data Ingestion

- [ ] **DATA-01**: Script can read the 30-pattern training set from a file or internal structure.
- [ ] **DATA-02**: Features $x_1, x_2, x_3$ are correctly mapped to NumPy arrays.
- [ ] **DATA-03**: Targets $d \in \{-1, 1\}$ are correctly mapped for supervised learning.

### Core Algorithm

- [ ] **ALGO-01**: Implementation of the Perceptron weighted sum $u = \sum w_i x_i$.
- [ ] **ALGO-02**: Inclusion of a bias term $x_0 = -1$.
- [ ] **ALGO-03**: Sign activation function: $y = 1$ if $u \ge 0$, else $-1$.
- [ ] **ALGO-04**: Supervised Hebbian Rule for weight updates: $w_{new} = w_{old} + \eta (d - y) x$.

### Training & Execution

- [ ] **TRAIN-01**: Support for 5 independent training runs.
- [ ] **TRAIN-02**: Randomized initial weights in the range $[0, 1]$ for each run.
- [ ] **TRAIN-03**: Termination when errors reach zero or after 1000 epochs.
- [ ] **TRAIN-04**: Reporting of results (final weights, number of epochs, final error count) for each run.

### Visualization

- [ ] **VIS-01**: Generate graphs at each epoch to visualize the decision boundary or error reduction.

- [ ] **REPORT-01**: Generate a web report (HTML/CSS) containing training results, classification tables, and theoretical answers.
- [ ] **TEST-01**: Classify 10 additional oil samples across all 5 training runs.

- **EVAL-01**: Performance visualization (error curve over epochs).
- **TEST-01**: Support for testing on a separate validation set.

## Out of Scope

| Feature | Reason |
|---------|--------|
| scikit-learn usage | Prohibited by university constraints. |
| GUI | CLI output is sufficient for research work. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Complete |
| DATA-03 | Phase 1 | Complete |
| ALGO-01 | Phase 2 | Complete |
| ALGO-02 | Phase 2 | Complete |
| ALGO-03 | Phase 2 | Complete |
| ALGO-04 | Phase 2 | Complete |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-02 | Phase 3 | Pending |
| TRAIN-03 | Phase 3 | Pending |
| TRAIN-04 | Phase 3 | Pending |
| VIS-01 | Phase 3 | Pending |
| REPORT-01 | Phase 3 | Pending |
| TEST-01 | Phase 3 | Pending |
| EVAL-01 | v2 | Pending |
| TEST-01-V2 | v2 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-30*
*Last updated: 2026-04-30 after initial definition*
