# Phase 3 — Execution & Output: Verification

## Status: PASS ✓

## Checklist

- [x] 5 treinamentos executados com seeds únicos [0, 42, 137, 256, 999]
- [x] Pesos iniciais distintos em cada run (seeds diferentes)
- [x] Pesos finais convergem para valores praticamente idênticos (confirmando solução estável)
- [x] Tabela impressa no terminal com w0–w4 iniciais, w0–w4 finais e épocas
- [x] 5 gráficos individuais gerados: `plots/convergencia_T1.png` … `T5.png`
- [x] 1 gráfico combinado: `plots/convergencia_todos.png`
- [x] `resultados.csv` gerado com tabela completa
- [x] Acurácia de 91.4% em todos os runs (3 padrões mal classificados dos 35)

## Resultados

### Pesos Iniciais

| Treino | w0_ini | w1_ini | w2_ini | w3_ini | w4_ini |
|--------|--------|--------|--------|--------|--------|
| T1     | 0.5488 | 0.7152 | 0.6028 | 0.5449 | 0.4237 |
| T2     | 0.3745 | 0.9507 | 0.7320 | 0.5987 | 0.1560 |
| T3     | 0.9439 | 0.0885 | 0.6954 | 0.7000 | 0.4996 |
| T4     | 0.0458 | 0.5861 | 0.2032 | 0.0842 | 0.0260 |
| T5     | 0.8034 | 0.5275 | 0.1191 | 0.6397 | 0.0909 |

### Pesos Finais + Épocas

| Treino | w0_fin | w1_fin | w2_fin | w3_fin | w4_fin | Épocas | Acurácia |
|--------|--------|--------|--------|--------|--------|--------|----------|
| T1     | 1.8112 | 1.3125 | 1.6413 | -0.4264 | -1.1771 | 831 | 91.4% |
| T2     | 1.8112 | 1.3126 | 1.6414 | -0.4263 | -1.1771 | 832 | 91.4% |
| T3     | 1.8113 | 1.3125 | 1.6413 | -0.4265 | -1.1771 | 817 | 91.4% |
| T4     | 1.8113 | 1.3125 | 1.6413 | -0.4266 | -1.1771 | 848 | 91.4% |
| T5     | 1.8113 | 1.3125 | 1.6413 | -0.4265 | -1.1771 | 829 | 91.4% |

## Observações

- Os pesos finais convergem para praticamente o mesmo vetor independente dos pesos iniciais — comportamento esperado para ADALINE em problema convexo (MSE é quadrático).
- Variação de épocas: 817–848 dependendo do ponto de partida.
- Acurácia de 91.4% (32/35 classificados corretamente) — dados ruidosos; esperado para ADALINE linear.

## Arquivos gerados

- `adaline/train.py`
- `adaline/resultados.csv`
- `adaline/plots/convergencia_T1.png` … `convergencia_T5.png`
- `adaline/plots/convergencia_todos.png`

---
*Verificado: 2026-05-07*
