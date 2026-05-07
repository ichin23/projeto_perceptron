# ADALINE Valve Signal Classifier

## What This Is

A Python-based implementation of an ADALINE (Adaptive Linear Neuron) network to classify noisy industrial control signals and route them to the correct valve (A or B). The system processes 4-dimensional input signals {x1, x2, x3, x4} transmitted over 500m with noise interference, and outputs -1 (Valve A) or +1 (Valve B). This is a university project focused on the Widrow-Hoff (LMS) learning rule.

## Core Value

Accurately classify noisy valve-routing signals using a manual ADALINE implementation, filling the results table with initial weights, final weights, and epoch counts for 5 independent training runs.

## Requirements

### Validated

- ✓ Implementation of ADALINE using the Widrow-Hoff (LMS/Delta) rule. — v1.0
- ✓ Support for 5 independent training runs with randomized initial weights [0, 1]. — v1.0
- ✓ Different random seed per run so initial weights are never repeated. — v1.0
- ✓ Integration of a bias term w0 (with fixed input x0 = 1). — v1.0
- ✓ Training with learning rate η = 0.0025. — v1.0
- ✓ Stopping criterion: change in MSE < ε = 10⁻⁶ (or equivalent precision condition). — v1.0
- ✓ Data ingestion from the 35-pattern training set. — v1.0
- ✓ Output table: for each run — initial weights (w0–w4), final weights (w0–w4), number of epochs. — v1.0
- ✓ Convergence graph (MSE vs. epochs) for each training run. — v1.0

### Active

(None currently — milestone completed)

### Out of Scope

- Use of `scikit-learn` or other high-level ML frameworks.
- Graphical User Interface (CLI only).
- Multi-layer or non-linear neural network architectures.
- Real-time signal acquisition from hardware.

## Context

- **Problem**: Two valves (A, B) share a single 500m communication channel. A switch near the valves needs to route signals correctly despite noise interference.
- **Dataset**: 35 patterns with 4 features (x1–x4) and labels d ∈ {-1, +1}. d=-1 → Valve A, d=+1 → Valve B.
- **Network structure**: 4 inputs + 1 bias → 1 linear output → sign activation for classification.
- **Learning rule**: Widrow-Hoff (LMS / Delta rule): Δw = η · (d - ŷ) · x
- **Stopping criterion**: |MSE(epoch) - MSE(epoch-1)| < ε = 10⁻⁶
- **Learning rate**: η = 0.0025
- **Environment**: Linux, Python 3.

**Current State (v1.0):**
Shipped v1.0 with 416 LOC in Python.
Tech stack: Python, NumPy, Pandas, Matplotlib.
Results show stable convergence around ~800-850 epochs with 91.4% accuracy across all initial random states.

## Constraints

- **Tech Stack**: Python 3 and NumPy (+ Matplotlib, Pandas allowed).
- **Algorithm**: Specifically the Widrow-Hoff (LMS) rule for ADALINE.
- **Initial Weights**: Random values in [0, 1] with a different seed per run.
- **Bias**: Input x0 = 1 (constant), weight w0 is trainable.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| MSE-delta stopping criterion | Matches ε = 10⁻⁶ precision requirement from assignment | ✓ Good |
| NumPy for math | Efficient vector operations, consistent with Perceptron project | ✓ Good |
| Sign activation for classification | ADALINE linear output → sign gives -1/+1 class label | ✓ Good |
| Separate random seed per run | Assignment requires non-repeated initial weights across runs | ✓ Good |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-07 after v1.0 milestone*
