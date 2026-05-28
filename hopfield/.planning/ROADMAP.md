# Roadmap

This document outlines the phased plan to implement the Hopfield Network Associative Memory project.

## Milestone v1.0: Hopfield Associative Memory

| Phase | Name | Goal | Requirements | Status |
|-------|------|------|--------------|--------|
| 1 | Core Hopfield & Tests | Implement Hopfield network logic and verify correctness with unit tests | HOP-01, HOP-02, HOP-03 | Pending |
| 2 | Simulation Suite | Implement noise simulation harness and gather experiment data | HOP-04, HOP-05 | Pending |
| 3 | Report & Integration | Generate interactive dashboard and integrate into main portfolio page | HOP-06, HOP-07 | Pending |

---

### Phase 1: Core Hopfield & Tests
Goal: Implement the core mathematical class for the 45-neuron Hopfield network and verify stability with unit tests.

#### Success Criteria
- [ ] Core patterns for digits 1, 2, 3, 4 defined as bipolar vectors of size 45 in `patterns.py`.
- [ ] Weight matrix constructed according to Hebbian learning rules, with zero diagonal.
- [ ] Asynchronous updates converge to stable states, and network energy decreases monotonically.
- [ ] Pytest suite passes all test assertions.

---

### Phase 2: Simulation Suite
Goal: Simulate noisy transmissions, verify recovery rates, and collect metrics.

#### Success Criteria
- [ ] Noise injection function flips a specified percentage of pixels randomly.
- [ ] Simulation runs 1000 trials at 20% noise to calculate recovery rate.
- [ ] Simulation runs sensitivity analysis (0% to 50% noise) and outputs results to CSV.
- [ ] Matplotlib plots showing success rate vs. noise level and energy logs saved in `hopfield/plots/`.

---

### Phase 3: Report & Integration
Goal: Create a visual glassmorphism report and connect it to the main portfolio.

#### Success Criteria
- [ ] `generate_report.py` builds `report.html` (and copy to `index.html` for routing) containing visual recovery comparison grids (Original -> Noisy -> Recovered).
- [ ] HTML report displays interactive Plotly charts showing success rate vs. noise level.
- [ ] Root `index.html` contains card for Hopfield project conforming to Outfit typography and background gradients.
- [ ] Root `README.md` updated with the Hopfield description.
