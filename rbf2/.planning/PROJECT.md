# RBF Gasoline Injection System Estimator (rbf2)

## What This Is

A Python-based implementation of a Radial Basis Function (RBF) neural network to map three input variables $\{x_1, x_2, x_3\}$ to a single output $\{y\}$, representing the amount of gasoline to be injected by an electronic fuel injection system. This is a non-linear regression/functional approximation problem. We will implement, train, and contrast three RBF topologies:
- **Rede 1:** $N_1 = 5$ hidden units
- **Rede 2:** $N_1 = 10$ hidden units
- **Rede 3:** $N_1 = 15$ hidden units

We will contrast the results of these models, comparing them with each other (and eventually with MLP results if provided).

## Core Value

Implement and train three RBF architectures to accurately map fuel injection parameters, comparing their approximation errors and convergence characteristics to identify the optimal configuration.

## Requirements

### Validated

*(None yet — starting from scratch)*

### Active

- [ ] **REQ-01**: Ingest training and test datasets once provided by the user.
- [ ] **REQ-02**: Implement the RBF neural network architecture with 3 inputs, $N_1$ hidden radial units, and 1 linear output unit.
- [ ] **REQ-03**: Implement center selection (e.g., K-Means clustering or random selection from training set) and spread ($\sigma$) computation.
- [ ] **REQ-04**: Implement output weights training (e.g., pseudo-inverse/least squares or Delta rule/LMS).
- [ ] **REQ-05**: Train all three candidate topologies ($N_1 \in \{5, 10, 15\}$) on the training set.
- [ ] **REQ-06**: Validate and compare the performance of the three topologies on the test set, computing MSE/RMSE and showing approximation plots.
- [ ] **REQ-07**: Generate a complete HTML/visual report summarizing the training process, center locations, weights, errors, and comparison.

### Out of Scope

- Use of high-level machine learning libraries (like scikit-learn, PyTorch, Keras) for the RBF implementation or training. Pure implementation in NumPy/Python is required.
- Graphical User Interface.

## Context

- **Problem**: Functional approximation of gasoline injection quantity $\{y\}$ from three vehicle variables $\{x_1, x_2, x_3\}$.
- **Dataset**: Awaiting training and testing dataset from the user.
- **Topologies**: 3 inputs, hidden layer with $N_1 \in \{5, 10, 15\}$ radial units, 1 output.
- **Environment**: Linux, Python 3.

**Current State (v0.1):**
Initializing project folder and waiting for dataset and specific questions from the user.

## Constraints

- **Tech Stack**: Python 3, NumPy, standard plotting/analysis libraries (Matplotlib, Pandas).
- **Algorithm**: Radial Basis Function (RBF) neural network.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| NumPy for implementation | Standard scientific library in Python, consistent with previous projects | ✓ Planned |

## Evolution

This document evolves at phase transitions and milestone boundaries.

---
*Last updated: 2026-05-20 after initialization*
