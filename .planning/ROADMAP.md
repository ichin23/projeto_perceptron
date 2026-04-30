# ROADMAP: Perceptron Oil Purity Classifier

## Milestone: Initial Implementation

Goal: Build a functional Perceptron using the supervised Hebbian rule and run 5 training trials as specified.

| # | Phase | Goal | Requirements | Status |
|---|-------|------|--------------|--------|
| 1 | Setup & Data | Prepare environment and training data file. | DATA-01, DATA-02, DATA-03 | Complete (2026-04-30) |
| 2 | Algorithm | Implement the Perceptron and Hebbian update rule. | ALGO-01, ALGO-02, ALGO-03, ALGO-04 | Pending |
| 3 | Execution | Implement the 5-run training loop, report results, and generate graphs. | TRAIN-01, TRAIN-02, TRAIN-03, TRAIN-04, VIS-01 | Pending |

## Phase Details

### Phase 1: Setup & Data
Prepare the project structure and convert the provided 30-pattern training set into a format readable by NumPy.
- **Success Criteria**:
  - `data.npy` or `data.csv` exists.
  - Test script can load data and print the shape (30, 4).

### Phase 2: Algorithm
Implement the core `Perceptron` class with the specified parameters ($\eta = 0.01$, bias $x_0 = -1$) and the Supervised Hebbian Rule.
- **Success Criteria**:
  - Unit test for `predict()` function.
  - Unit test for `update_weights()` function.

### Phase 3: Execution
Implement the logic to run the training 5 times, each starting with random weights $[0, 1]$, and stopping at zero error or 1000 epochs. Generate progress graphs at each epoch.
- **Success Criteria**:
  - Script output shows results for 5 independent runs.
  - Weights are randomized correctly between runs.
  - Graphs are generated and saved (or displayed) per epoch.

---
*Last updated: 2026-04-30 after initialization*
