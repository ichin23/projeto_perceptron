# Phase 1 — Setup & Data: Verification

## Status: PASS ✓

## Checklist

- [x] `training_data.csv` has shape (35, 5) — **verified by NumPy**
- [x] `load_data.py` returns `X_bias` shape (35, 5) and `d` shape (35,) — **verified**
- [x] Bias column (x0) is all 1.0 — **verified**
- [x] Labels set is {-1.0, 1.0} — **verified**
- [x] Class distribution: 18 patterns Valve A (d=-1), 17 patterns Valve B (d=+1)

## Output

```
X_bias shape: (35, 5)
d shape:      (35,)
Bias column (x0): [1. 1. 1.]
First labels: [ 1. -1. -1. -1.  1.]
Class A (Valve A, d=-1): 18 patterns
Class B (Valve B, d=+1): 17 patterns
```

## Files Created

- `adaline/training_data.csv` — 35-pattern training dataset
- `adaline/load_data.py` — data loader with bias prepending

---
*Verified: 2026-05-07*
