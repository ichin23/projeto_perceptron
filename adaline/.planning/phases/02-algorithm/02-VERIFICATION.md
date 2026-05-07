# Phase 2 — Algorithm: Verification

## Status: PASS ✓

## Checklist

- [x] `adaline.py` implementa regra online de Widrow-Hoff: `Δw = η·(d - wᵀx)·x`
- [x] Ativação linear durante treino (`net = wᵀx`); função `sign` para classificação
- [x] Critério de parada: `|MSE(t) - MSE(t-1)| < ε = 1e-6` — verificado (Δ final = 9.97e-7)
- [x] Bias weight w0 treinável (input x0=1 já em X_bias desde Fase 1)
- [x] Todos os 7 testes unitários passaram

## Saída dos testes

```
Executando testes da ADALINE...

  [OK] test_predict_shape
  [OK] test_predict_values
  [OK] test_train_returns_epoch — convergiu em 832 épocas
  [OK] test_weights_shape
  [OK] test_convergence_criterion — Δ MSE final = 9.97e-07
  [OK] test_bias_used — w0 = 1.811230
  [OK] test_mse_history_populated — 851 épocas registradas

========================================
Todos os 7 testes passaram.
```

## Arquivos criados

- `adaline/adaline.py` — classe ADALINE com regra LMS e critério de parada
- `adaline/test_adaline.py` — 7 testes unitários

## Observações

- Convergência típica: ~800–900 épocas com η=0.0025, ε=1e-6
- Peso w0 (bias) significativo (~1.8), confirmando importância do termo de bias

---
*Verificado: 2026-05-07*
