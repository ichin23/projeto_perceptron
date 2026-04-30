# Perceptron Oil Purity Classifier

## What This Is

A Python-based implementation of a Perceptron neural network to classify oil purity into two classes (C1 and C2) based on three physico-chemical properties. This is a university project focusing on supervised learning using the Hebbian rule.

## Core Value

Accurately classify oil purity patterns using a manual Perceptron implementation without high-level libraries like scikit-learn.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Implementation of Perceptron using the Supervised Hebbian Rule.
- [ ] Support for 5 independent training runs with randomized initial weights.
- [ ] Integration of a bias term $x_0 = -1$.
- [ ] Training termination based on zero errors or a 1000-epoch limit.
- [ ] Use of NumPy for vector operations.
- [ ] Data ingestion from the provided 30-pattern training set.

### Out of Scope

- Use of `scikit-learn` or other high-level ML frameworks.
- Graphical User Interface (CLI only).
- Advanced neural network architectures (Multi-layer Perceptrons, etc.).

## Context

- **Dataset**: 30 patterns with 3 features ($x_1, x_2, x_3$) and labels $d \in \{-1, 1\}$.
- **Classes**: C1 ($d = -1$), C2 ($d = 1$).
- **Learning Rate**: $\eta = 0.01$.
- **Environment**: Linux, Python 3.

## Constraints

- **Tech Stack**: Python 3 and NumPy only.
- **Algorithm**: Specifically the Supervised Hebbian Rule for Perceptron.
- **Initial Weights**: Must be random values between 0 and 1.
- **Bias**: Fixed input $x_0 = -1$.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| NumPy for math | Efficient vector operations as per project allowance. | — Pending |
| Sign function for activation | Standard for binary classification with targets -1 and 1. | — Pending |

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
*Last updated: 2026-04-30 after initialization*
