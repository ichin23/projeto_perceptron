# Hopfield Network Associative Memory

## What This Is

A Python-based implementation of a 45-neuron Hopfield Network to store and retrieve four 9x5 pixel binary patterns (digits "1", "2", "3", and "4") from corrupted versions (e.g., 20% random noise). The project includes a simulation suite to analyze network capacity and recovery rates, unit tests, and a dynamic HTML report dashboard that visualizes pattern reconstruction and convergence.

## Core Value

Demonstrate the behavior of associative memories under noise, detailing pattern convergence, recovery rates across different noise thresholds (0% to 50%), and the network's energy minimization trajectory.

## Requirements

### Active

(None currently — milestone completed)

### Validated

- ✓ **HOP-01**: Implement a Hopfield Network model with 45 neurons. — v1.0
- ✓ **HOP-02**: Calculate the weight matrix using Hebbian learning for the 4 target patterns (digits 1, 2, 3, 4). — v1.0
- ✓ **HOP-03**: Support asynchronous state update dynamics and track the network's energy function. — v1.0
- ✓ **HOP-04**: Simulate image transmission over a noisy communication link (20% random pixel corruption) and evaluate recovery. — v1.0
- ✓ **HOP-05**: Generate an interactive HTML report dashboard (`report.html`) summarizing the experiment, showcasing visual grid transformations (Original -> Noisy -> Recovered) and plotting recovery rate vs. noise level. — v1.0
- ✓ **HOP-06**: Integrate the Hopfield project card into the portfolio home page. — v1.0

### Out of Scope

- Use of external machine learning libraries (like `scikit-learn`, `PyTorch`, etc.) for network implementation.
- Real-time video/image processing of external files (inputs are predefined 9x5 grids).
- GUI applications outside of the static HTML/CSS/JS interactive dashboard.

## Context

- **Grid Size**: 9 rows by 5 columns = 45 pixels / neurons.
- **Pixel Encoding**: Bipolar values: White pixel = -1, Dark/Grey pixel = +1.
- **Noise Model**: 20% random pixel corruption (i.e. exactly 9 out of 45 pixels are flipped in state during transmission).
- **Update Dynamics**: Asynchronous updates (neurons updated in random order permutations until convergence) to guarantee energy minimization and stable convergence.
- **Visuals**: Modern responsive HTML report using glassmorphism styling, Plotly charts, and grid layout representations.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Bipolar State Coding | Recommended by prompt (-1/1) which provides better noise handling and symmetric weights in Hopfield networks | ✓ Approved |
| Asynchronous Update | Ensures convergence to local energy minima and avoids limit cycles (2-period oscillations) common in synchronous updates | ✓ Approved |
| Interactive Plotly Dashboard | Consistent with portfolio visual requirements; allows interactive inspection of simulation results | ✓ Approved |

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
*Last updated: 2026-05-28 after initialization*
