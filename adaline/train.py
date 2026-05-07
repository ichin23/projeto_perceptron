"""
train.py — Execução dos 5 treinamentos da rede ADALINE

Classificador de sinais de válvulas industriais ruidosos.
  d = -1 → Válvula A
  d = +1 → Válvula B

Parâmetros:
  η (eta)     = 0.0025
  ε (epsilon) = 1e-6  (critério de parada: |ΔMSE| < ε)
  5 runs com seeds únicos: pesos iniciais aleatórios em [0, 1]

Saída:
  - Tabela de resultados no terminal (pandas DataFrame)
  - Gráficos PNG em adaline/plots/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # backend sem display (salva para arquivo)

from load_data import load_data
from adaline import ADALINE

# ─── Configurações ────────────────────────────────────────────────────────────
ETA     = 0.0025
EPSILON = 1e-6
N_RUNS  = 5
SEEDS   = [0, 42, 137, 256, 999]   # um seed diferente por run
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Carregar dados ────────────────────────────────────────────────────────────
X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))

# ─── Executar 5 treinamentos ──────────────────────────────────────────────────
results = []

print("=" * 70)
print("  ADALINE — Classificador de Sinais de Válvulas Industriais")
print(f"  η = {ETA}  |  ε = {EPSILON}  |  Dataset: {X.shape[0]} padrões")
print("=" * 70)

for run in range(1, N_RUNS + 1):
    seed = SEEDS[run - 1]
    np.random.seed(seed)

    # Pesos iniciais aleatórios em [0, 1]
    w_initial = np.random.uniform(0, 1, X.shape[1])

    # Criar e treinar ADALINE
    model = ADALINE(eta=ETA, epsilon=EPSILON)
    model.set_weights(w_initial)
    epochs = model.train(X, d)

    w_final = model.weights.copy()
    acc = model.accuracy(X, d)

    print(f"\nTreinamento T{run} (seed={seed})")
    print(f"  Pesos iniciais: {np.round(w_initial, 4)}")
    print(f"  Pesos finais:   {np.round(w_final, 4)}")
    print(f"  Épocas:         {epochs}")
    print(f"  Acurácia:       {acc * 100:.1f}%")

    results.append({
        'Treino': f'T{run}',
        'w0_ini': round(float(w_initial[0]), 4),
        'w1_ini': round(float(w_initial[1]), 4),
        'w2_ini': round(float(w_initial[2]), 4),
        'w3_ini': round(float(w_initial[3]), 4),
        'w4_ini': round(float(w_initial[4]), 4),
        'w0_fin': round(float(w_final[0]), 4),
        'w1_fin': round(float(w_final[1]), 4),
        'w2_fin': round(float(w_final[2]), 4),
        'w3_fin': round(float(w_final[3]), 4),
        'w4_fin': round(float(w_final[4]), 4),
        'Épocas': epochs,
        'Acurácia': f'{acc * 100:.1f}%',
        '_mse_history': model.mse_history,  # para os gráficos (excluído do df)
    })

    # ── Gráfico individual de convergência ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(model.mse_history) + 1), model.mse_history,
            color='#2196F3', linewidth=1.2, label=f'MSE - T{run}')
    ax.set_xlabel('Época', fontsize=11)
    ax.set_ylabel('MSE (Erro Quadrático Médio)', fontsize=11)
    ax.set_title(f'Convergência ADALINE — Treinamento T{run} '
                 f'(seed={seed}, {epochs} épocas)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    path_individual = os.path.join(PLOTS_DIR, f'convergencia_T{run}.png')
    fig.savefig(path_individual, dpi=150)
    plt.close(fig)
    print(f"  Gráfico salvo: {path_individual}")

# ─── Tabela de resultados ─────────────────────────────────────────────────────
mse_histories = [r.pop('_mse_history') for r in results]  # separar antes do df

df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("  TABELA DE RESULTADOS")
print("=" * 70)

# Separar em duas tabelas para melhor legibilidade
cols_ini = ['Treino', 'w0_ini', 'w1_ini', 'w2_ini', 'w3_ini', 'w4_ini']
cols_fin = ['Treino', 'w0_fin', 'w1_fin', 'w2_fin', 'w3_fin', 'w4_fin', 'Épocas', 'Acurácia']

print("\nPesos Iniciais:")
print(df[cols_ini].to_string(index=False))
print("\nPesos Finais + Épocas:")
print(df[cols_fin].to_string(index=False))

# Salvar tabela completa em CSV
csv_path = os.path.join(os.path.dirname(__file__), 'resultados.csv')
df.to_csv(csv_path, index=False)
print(f"\nTabela salva em: {csv_path}")

# ─── Gráfico combinado de todos os runs ───────────────────────────────────────
colors = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0']

fig, ax = plt.subplots(figsize=(10, 6))
for i, (hist, run_info) in enumerate(zip(mse_histories, results)):
    label = f"T{i+1} ({run_info['Épocas']} épocas)"
    ax.plot(range(1, len(hist) + 1), hist,
            color=colors[i], linewidth=1.3, label=label, alpha=0.85)

ax.set_xlabel('Época', fontsize=12)
ax.set_ylabel('MSE (Erro Quadrático Médio)', fontsize=12)
ax.set_title('Convergência ADALINE — Todos os Treinamentos\n'
             f'η = {ETA}  |  ε = {EPSILON}', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
fig.tight_layout()
path_combined = os.path.join(PLOTS_DIR, 'convergencia_todos.png')
fig.savefig(path_combined, dpi=150)
plt.close(fig)
print(f"Gráfico combinado salvo: {path_combined}")

print("\n" + "=" * 70)
print("  Execução concluída.")
print("=" * 70)
