# Requirements

This document tracks the requirements for the Hopfield Network Associative Memory project.

## Milestone v1.0: Hopfield Associative Memory

### Core Network Implementation
- [ ] **HOP-01**: The system must implement a Hopfield network of exactly 45 neurons.
- [ ] **HOP-02**: The network weights must be trained/calculated using Hebbian learning over the 4 target patterns (digits 1, 2, 3, 4). Weight diagonal must be 0 ($w_{ii} = 0$).
- [ ] **HOP-03**: The update rule must support asynchronous state updates using random permutations of neuron indices. The system must track the state energy $E = -\frac{1}{2}\sum w_{ij}s_is_j$ and verify that it decreases or stays constant on each step.

### Simulation and Verification
- [ ] **HOP-04**: The system must simulate a transmission link with 20% random noise corruption (exactly 9 pixels flipped) and evaluate recovery rate over 1000 trials per pattern.
- [ ] **HOP-05**: The system must run a sensitivity analysis by varying noise level from 0% to 50% in steps of 5%, recording recovery success rates and average convergence epochs.

### Visualization and Portfolio
- [ ] **HOP-06**: The system must generate an interactive HTML report dashboard (`report.html` and `index.html`) featuring:
  - Responsive glassmorphism layout matching the portfolio design.
  - Interactive grid representations showing: Original Pattern, Noisy Pattern (20% noise), and Reconstructed Pattern.
  - Interactive Plotly chart showing Recovery Rate vs. Noise Level.
  - Convergence analysis detailing energy trajectory and epoch counts.
- [ ] **HOP-07**: The main portfolio `index.html` and root `README.md` must be updated to include cards and links for the Hopfield project.

## Future Requirements (Deferred)
- [ ] **HOP-08**: Multi-class pattern classifier addition.
- [ ] **HOP-09**: Continuous state Hopfield models.

## Out of Scope
- Real-time video feeding or external image upload (predefined 9x5 matrices only).
- Standard machine learning libraries (implement network from scratch).

## Traceability
| Req ID | Phase | Test Case | Status |
|--------|-------|-----------|--------|
| HOP-01 | Phase 1 | `test_hopfield.py:test_dimensions` | Pending |
| HOP-02 | Phase 1 | `test_hopfield.py:test_hebbian_weights` | Pending |
| HOP-03 | Phase 1 | `test_hopfield.py:test_energy_minimization` | Pending |
| HOP-04 | Phase 2 | `simulation.py` execution | Pending |
| HOP-05 | Phase 2 | `simulation.py` execution | Pending |
| HOP-06 | Phase 3 | `generate_report.py` execution | Pending |
| HOP-07 | Phase 3 | Manual check of main page | Pending |
