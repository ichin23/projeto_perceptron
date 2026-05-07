# REQUIREMENTS: ADALINE Valve Signal Classifier

## Milestone 1: Initial Implementation

### Functional Requirements

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| ALGO-01 | Implement ADALINE class with Widrow-Hoff (LMS) learning rule: Δw = η·(d - ŷ)·x | Must | Assignment |
| ALGO-02 | Linear activation (net = wᵀx) for training; sign function for final classification | Must | Assignment |
| ALGO-03 | Stopping criterion: change in MSE between consecutive epochs < ε = 10⁻⁶ | Must | Assignment |
| ALGO-04 | Bias term: fixed input x0 = 1, trainable weight w0 | Must | Assignment |
| DATA-01 | Ingest 35-pattern training set (x1, x2, x3, x4, d) from CSV file | Must | Assignment |
| DATA-02 | Prepend bias column x0 = 1 to each input vector | Must | Assignment |
| TRAIN-01 | Run 5 independent training sessions with η = 0.0025 | Must | Assignment |
| TRAIN-02 | Initialize weights randomly from uniform [0, 1] with unique seed per run | Must | Assignment |
| TRAIN-03 | Record initial weights (w0–w4) per run | Must | Assignment |
| TRAIN-04 | Record final weights (w0–w4) per run | Must | Assignment |
| TRAIN-05 | Record number of epochs until convergence per run | Must | Assignment |
| OUT-01 | Print results table: run number, initial weights, final weights, epochs | Must | Assignment |
| OUT-02 | Generate MSE convergence graph per training run | Should | Assignment |

### Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-01 | Python 3 + NumPy only (no sklearn, tensorflow, etc.) |
| NFR-02 | Matplotlib allowed for graphs |
| NFR-03 | Code must be reproducible: same seeds → same initial weights |

---
*Last updated: 2026-05-07 after initialization*
