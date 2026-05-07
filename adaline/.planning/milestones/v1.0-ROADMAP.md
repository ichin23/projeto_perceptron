# ROADMAP: ADALINE Valve Signal Classifier

## Milestone 1: Initial Implementation

Goal: Build a functional ADALINE network using the Widrow-Hoff rule, run 5 training trials, and produce the required results table and convergence graphs.

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Setup & Data | Prepare project structure and training data CSV. | DATA-01, DATA-02 | Planned |
| 2 | Algorithm | Implement ADALINE class with LMS rule and stopping criterion. | ALGO-01, ALGO-02, ALGO-03, ALGO-04 | Planned |
| 3 | Execution & Output | Run 5 training sessions, produce results table and convergence graphs. | TRAIN-01..05, OUT-01, OUT-02 | Planned |

## Phase Details

### Phase 1: Setup & Data
Prepare the project structure and save the 35-pattern training set as a CSV file readable by NumPy/pandas.
- **Success Criteria**:
  - `training_data.csv` exists with 35 rows and columns x1, x2, x3, x4, d.
  - A load script can read and print `shape = (35, 5)`.

### Phase 2: Algorithm
Implement the `ADALINE` class with Widrow-Hoff (LMS) update rule, linear activation during training, sign activation for classification, bias term, and MSE-delta stopping criterion.
- **Success Criteria**:
  - Unit test: `predict()` returns a scalar for a given weight vector and input.
  - Unit test: `train()` converges on a simple linearly separable dataset.
  - Unit test: stopping triggers when |ΔMSE| < 10⁻⁶.

### Phase 3: Execution & Output
Implement the training loop for 5 independent runs. Each run uses a unique random seed for initial weights in [0, 1]. Output the results table and MSE convergence graphs.
- **Success Criteria**:
  - Script prints a table with 5 rows: run #, w0–w4 initial, w0–w4 final, epochs.
  - 5 convergence plots saved as PNG files (or displayed).
  - Initial weights differ across runs (different seeds verified).

---
*Last updated: 2026-05-07 after initialization*
