# RBF Radiation Classifier

## What This Is

A Python-based implementation of a Radial Basis Function (RBF) neural network to classify the presence or absence of radiation in nuclear compounds. The system processes a 2-dimensional input space representing concentration variables {x1, x2} and outputs a classification: +1 (presence of radiation) or -1 (absence of radiation). This is a university project based on the textbook "Redes Neurais Artificiais para Engenharia e Ciências Aplicadas".

## Core Value

Design and train an RBF neural network (2 inputs, 1 output) capable of accurately classifying radiation presence using training and test data, exploring clustering for centers (like K-means or selection) and calculating performance.

## Requirements

### Validated

(None yet — starting from scratch)

### Active

- [ ] Set up project directory and ingestion of the training dataset.
- [ ] Implement RBF neural network architecture (Input layer, Hidden/Radial layer, Output layer).
- [ ] Establish methods for choosing RBF centers (e.g. random selection from training set, or clustering) and standard deviation (spread parameter).
- [ ] Train the output layer weights using the Least Squares method or Gradient Descent.
- [ ] Validate the trained RBF network on test/validation patterns.
- [ ] Generate convergence/error plots and classification performance metrics.

### Out of Scope

- Use of high-level machine learning libraries for RBF training (e.g., scikit-learn's RBF-like methods or MLP classifiers). Pure implementation using NumPy/Python is required.
- Graphical User Interface.

## Context

- **Problem**: Classify if a nuclear compound contains radiation (y = 1) or not (y = -1) based on concentration features x1 and x2.
- **Dataset**: A total of 50 situations (usually divided into training and testing sets, to be provided by the user).
- **Network structure**: 2 inputs (x1, x2), a hidden layer of radial basis functions, and 1 output (y).
- **Environment**: Linux, Python 3.

**Current State (v0.1):**
Initializing project folder and waiting for dataset input.

## Constraints

- **Tech Stack**: Python 3, NumPy, and standard plotting/analysis libraries (Matplotlib, Pandas).
- **Algorithm**: Radial Basis Function (RBF) neural network.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| NumPy for implementation | Standard scientific library in Python, consistent with previous projects | ✓ Planned |

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
*Last updated: 2026-05-20 after initialization*
