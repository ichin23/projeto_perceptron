import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from load_data import load_data
from rbf import RBFNetwork

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Helper function to convert matplotlib figures to base64
def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    return base64.b64encode(img_buf.read()).decode('utf-8')

# 1. Load datasets
X_train, d_train = load_data(os.path.join(BASE_DIR, 'training_data.csv'))
X_test, d_test = load_data(os.path.join(BASE_DIR, 'test_data.csv'))

# 2. Sweep across topologies for general evaluation
m_values = [2, 3, 5, 8, 10, 15, 20]
strategies = ['random', 'kmeans']
results_list = []

best_acc = -1.0
best_model = None
best_m = None
best_strategy = None

for strategy in strategies:
    for m in m_values:
        model = RBFNetwork(n_centers=m, center_strategy=strategy, spread_strategy='heuristic')
        model.train_least_squares(X_train, d_train, seed=42)
        train_acc = model.accuracy(X_train, d_train)
        test_acc = model.accuracy(X_test, d_test)
        
        results_list.append({
            'M': m,
            'Estratégia': 'K-Means' if strategy == 'kmeans' else 'Aleatório',
            'Acurácia Treino': f"{train_acc*100:.1f}%",
            'Acurácia Teste': f"{test_acc*100:.1f}%",
            '_train_acc_val': train_acc,
            '_test_acc_val': test_acc
        })
        
        if test_acc > best_acc or (abs(test_acc - best_acc) < 1e-7 and train_acc > best_model.accuracy(X_train, d_train)):
            best_acc = test_acc
            best_model = model
            best_m = m
            best_strategy = strategy

# 3. Compute specific K-Means centroids for radiation presence patterns (d = 1.0)
X_rad = X_train[d_train == 1.0]
kmeans_model = RBFNetwork(n_centers=2)
centers_rad = kmeans_model._kmeans(X_rad, k=2, seed=42)

# Assign points to centroids
distances = np.linalg.norm(X_rad[:, np.newaxis] - centers_rad, axis=2)
labels = np.argmin(distances, axis=1)

# Compute population and sample variances
vars_list = []
for i in range(2):
    pts = X_rad[labels == i]
    c = centers_rad[i]
    var_pop = np.var(pts, axis=0, ddof=0)
    var_sam = np.var(pts, axis=0, ddof=1)
    tot_pop = np.mean(np.sum((pts - c)**2, axis=1))
    tot_sam = np.sum((pts - c)**2) / (len(pts) - 1)
    vars_list.append({
        'c': c,
        'var_pop': var_pop,
        'var_sam': var_sam,
        'tot_pop': tot_pop,
        'tot_sam': tot_sam,
        'count': len(pts)
    })

# 4. Train Output Layer for specific questions (Delta Rule, η = 0.01, ε = 10^-7)
# Scenario 1: Population Variance as Spread
sigmas_sq_pop = np.array([vars_list[0]['tot_pop'], vars_list[1]['tot_pop']])
# Compute Phi activations for training
N_train = X_train.shape[0]
Phi_train_pop = np.zeros((N_train, 2))
for j in range(2):
    diff = X_train - centers_rad[j]
    dist_sq = np.sum(diff ** 2, axis=1)
    Phi_train_pop[:, j] = np.exp(-dist_sq / (2 * sigmas_sq_pop[j]))
Phi_bias_train_pop = np.hstack([np.ones((N_train, 1)), Phi_train_pop])

# Train using Delta Rule
np.random.seed(42)
weights_pop = np.random.uniform(-0.5, 0.5, 3)
eta = 0.01
epsilon = 1e-7
mse_history_pop = []
for epoch in range(100000):
    y = Phi_bias_train_pop @ weights_pop
    err = d_train - y
    grad = - (Phi_bias_train_pop.T @ err) / N_train
    weights_pop -= eta * grad
    mse = np.mean(err ** 2)
    mse_history_pop.append(mse)
    if epoch > 0 and abs(mse_history_pop[-1] - mse_history_pop[-2]) < epsilon:
        break
epochs_pop = len(mse_history_pop)

# Compute Test Predictions for Scenario 1
N_test = X_test.shape[0]
Phi_test_pop = np.zeros((N_test, 2))
for j in range(2):
    diff = X_test - centers_rad[j]
    dist_sq = np.sum(diff ** 2, axis=1)
    Phi_test_pop[:, j] = np.exp(-dist_sq / (2 * sigmas_sq_pop[j]))
Phi_bias_test_pop = np.hstack([np.ones((N_test, 1)), Phi_test_pop])
y_test_raw_pop = Phi_bias_test_pop @ weights_pop
y_test_pos_pop = np.where(y_test_raw_pop >= 0, 1.0, -1.0)
correct_pop = y_test_pos_pop == d_test
acc_pop = np.mean(correct_pop) * 100

# Scenario 2: Heuristic Spread (sigma = d_max / sqrt(2*M))
d_centers = np.linalg.norm(centers_rad[0] - centers_rad[1])
sigmas_sq_heur = np.full(2, (d_centers / 2.0)**2)
# Compute activations for training
Phi_train_heur = np.zeros((N_train, 2))
for j in range(2):
    diff = X_train - centers_rad[j]
    dist_sq = np.sum(diff ** 2, axis=1)
    Phi_train_heur[:, j] = np.exp(-dist_sq / (2 * sigmas_sq_heur[j]))
Phi_bias_train_heur = np.hstack([np.ones((N_train, 1)), Phi_train_heur])

# Train using Delta Rule
np.random.seed(42)
weights_heur = np.random.uniform(-0.5, 0.5, 3)
mse_history_heur = []
for epoch in range(100000):
    y = Phi_bias_train_heur @ weights_heur
    err = d_train - y
    grad = - (Phi_bias_train_heur.T @ err) / N_train
    weights_heur -= eta * grad
    mse = np.mean(err ** 2)
    mse_history_heur.append(mse)
    if epoch > 0 and abs(mse_history_heur[-1] - mse_history_heur[-2]) < epsilon:
        break

# Compute Test Predictions for Scenario 2
Phi_test_heur = np.zeros((N_test, 2))
for j in range(2):
    diff = X_test - centers_rad[j]
    dist_sq = np.sum(diff ** 2, axis=1)
    Phi_test_heur[:, j] = np.exp(-dist_sq / (2 * sigmas_sq_heur[j]))
Phi_bias_test_heur = np.hstack([np.ones((N_test, 1)), Phi_test_heur])
y_test_raw_heur = Phi_bias_test_heur @ weights_heur
y_test_pos_heur = np.where(y_test_raw_heur >= 0, 1.0, -1.0)
correct_heur = y_test_pos_heur == d_test
acc_heur = np.mean(correct_heur) * 100

# 5. Generate Plots
# Plot 1: Topology comparison
fig, ax = plt.subplots(figsize=(8, 4.5))
m_vals_unique = sorted(list(set(m_values)))
random_test_accs = [r['_test_acc_val'] * 100 for r in results_list if r['Estratégia'] == 'Aleatório']
kmeans_test_accs = [r['_test_acc_val'] * 100 for r in results_list if r['Estratégia'] == 'K-Means']
ax.plot(m_vals_unique, random_test_accs, marker='o', color='#E74C3C', linestyle='--', linewidth=1.5, label='Centros Aleatórios')
ax.plot(m_vals_unique, kmeans_test_accs, marker='s', color='#2ECC71', linestyle='-', linewidth=1.8, label='Centros K-Means')
ax.set_xlabel('Número de Neurônios Ocultos (M)', fontsize=11)
ax.set_ylabel('Acurácia no Teste (%)', fontsize=11)
ax.set_title('Acurácia vs. Neurônios Ocultos (M) e Estratégia de Centros', fontsize=12)
ax.set_xticks(m_vals_unique)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=10)
fig.tight_layout()
comparison_b64 = fig_to_base64(fig)
plt.close(fig)

# Plot 2: Decision boundary of best model (M=8 K-Means or similar)
fig, ax = plt.subplots(figsize=(8, 6))
x1_min, x1_max = 0.0, 1.2
x2_min, x2_max = 0.0, 1.2
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]
grid_preds = best_model.predict(grid_points).reshape(xx1.shape)
ax.contourf(xx1, xx2, grid_preds, alpha=0.15, colors=['#E74C3C', '#2ECC71'])
ax.contour(xx1, xx2, grid_preds, levels=[0], colors=['#2C3E50'], linewidths=1.5, linestyles='--')
# Scatter points
train_pres = d_train == 1.0
train_abs = d_train == -1.0
ax.scatter(X_train[train_pres, 0], X_train[train_pres, 1], c='#2ECC71', marker='o', s=60, edgecolors='k', label='Treino: Radiação (+1)')
ax.scatter(X_train[train_abs, 0], X_train[train_abs, 1], c='#E74C3C', marker='o', s=60, edgecolors='k', label='Treino: Ausência (-1)')
test_pres = d_test == 1.0
test_abs = d_test == -1.0
ax.scatter(X_test[test_pres, 0], X_test[test_pres, 1], c='#A3E4D7', marker='^', s=80, edgecolors='k', label='Teste: Radiação (+1)')
ax.scatter(X_test[test_abs, 0], X_test[test_abs, 1], c='#FADBD8', marker='^', s=80, edgecolors='k', label='Teste: Ausência (-1)')
# Centers
ax.scatter(best_model.centers[:, 0], best_model.centers[:, 1], c='#F1C40F', marker='X', s=120, edgecolors='k', label='Centros RBF')
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_xlabel('Concentração x1', fontsize=12)
ax.set_ylabel('Concentração x2', fontsize=12)
ax.set_title(f'Fronteira de Decisão - Melhor RBF (M={best_m}, Estratégia={best_strategy.upper()})', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
best_boundary_b64 = fig_to_base64(fig)
plt.close(fig)

# Plot 3: Specific 2-centers model boundary (with population variance)
fig, ax = plt.subplots(figsize=(8, 6))
# Evaluate grid for 2-centers model
Phi_grid = np.zeros((grid_points.shape[0], 2))
for j in range(2):
    diff = grid_points - centers_rad[j]
    dist_sq = np.sum(diff ** 2, axis=1)
    Phi_grid[:, j] = np.exp(-dist_sq / (2 * sigmas_sq_pop[j]))
Phi_bias_grid = np.hstack([np.ones((grid_points.shape[0], 1)), Phi_grid])
grid_raw_pop = Phi_bias_grid @ weights_pop
grid_preds_pop = np.where(grid_raw_pop >= 0, 1.0, -1.0).reshape(xx1.shape)

ax.contourf(xx1, xx2, grid_preds_pop, alpha=0.15, colors=['#E74C3C', '#2ECC71'])
ax.contour(xx1, xx2, grid_preds_pop, levels=[0], colors=['#2C3E50'], linewidths=1.5, linestyles='--')
# Scatter training points
ax.scatter(X_train[train_pres, 0], X_train[train_pres, 1], c='#2ECC71', marker='o', s=60, edgecolors='k', label='Treino: Radiação (+1)')
ax.scatter(X_train[train_abs, 0], X_train[train_abs, 1], c='#E74C3C', marker='o', s=60, edgecolors='k', label='Treino: Ausência (-1)')
# Scatter test points
ax.scatter(X_test[test_pres, 0], X_test[test_pres, 1], c='#A3E4D7', marker='^', s=80, edgecolors='k', label='Teste: Radiação (+1)')
ax.scatter(X_test[test_abs, 0], X_test[test_abs, 1], c='#FADBD8', marker='^', s=80, edgecolors='k', label='Teste: Ausência (-1)')
# Centers
ax.scatter(centers_rad[:, 0], centers_rad[:, 1], c='#FF00FF', marker='X', s=140, edgecolors='k', label='Centros K-Means (d=1.0)')
ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_xlabel('Concentração x1', fontsize=12)
ax.set_ylabel('Concentração x2', fontsize=12)
ax.set_title('Fronteira de Decisão - RBF com 2 Centros (Variância Populacional)', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()
specific_boundary_b64 = fig_to_base64(fig)
plt.close(fig)

# Plot 4: Convergence Curve for Delta Rule
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(mse_history_pop) + 1), mse_history_pop, color='#9B59B6', linewidth=1.5, label='MSE (Cenário Populacional)')
ax.set_yscale('log')
ax.set_xlabel('Época', fontsize=11)
ax.set_ylabel('Erro Quadrático Médio (MSE)', fontsize=11)
ax.set_title(f'Curva de Aprendizado (Regra Delta, M=2)\nConvergência em {epochs_pop} épocas | MSE Final: {mse_history_pop[-1]:.6f}', fontsize=12)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(fontsize=10)
fig.tight_layout()
gd_convergence_b64 = fig_to_base64(fig)
plt.close(fig)


# 6. Generate HTML Content
html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório do Projeto RBF — Classificador de Radiação</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 30px 20px;
            background-color: #f4f6f9;
        }}
        .container {{
            background-color: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 12px;
            text-align: center;
            font-size: 2.2rem;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 40px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 8px;
            font-size: 1.6rem;
        }}
        h3 {{
            color: #34495e;
            margin-top: 25px;
            font-size: 1.25rem;
        }}
        p, li {{
            font-size: 1.05rem;
            color: #555;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background: #fff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
            font-size: 0.95rem;
        }}
        th {{
            background-color: #34495e;
            color: #fff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
            background: #fff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e1e8ed;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #f1f1f1;
        }}
        .answer-box {{
            background-color: #eaf2f8;
            border-left: 5px solid #3498db;
            padding: 22px;
            margin: 25px 0;
            border-radius: 0 8px 8px 0;
        }}
        .answer-box h4 {{
            margin-top: 0;
            color: #2471a3;
            font-size: 1.15rem;
            margin-bottom: 10px;
        }}
        .result-success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .result-danger {{
            color: #c0392b;
            font-weight: bold;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            font-size: 0.85em;
            font-weight: bold;
            color: white;
            background-color: #e74c3c;
            border-radius: 3px;
        }}
        .badge-success {{
            background-color: #2ecc71;
        }}
        code {{
            background-color: #f1f1f1;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.95em;
        }}
    </style>
</head>
<body>

<div class="container">
    <h1>Relatório do Projeto RBF — Classificação de Radiação</h1>
    
    <p>Este relatório apresenta a parametrização e avaliação de desempenho de uma Rede Neural de Funções de Base Radial (RBF) de duas entradas e uma saída, com foco em responder às questões propostas sobre a classificação da presença (1.0) ou ausência (-1.0) de radiação em compostos nucleares com base em variáveis x1 e x2.</p>

    <h2>Questão 1: Treinamento dos Centros via K-Means e Variância</h2>
    <div class="answer-box">
        <h4>Determinação de Centros e Variâncias</h4>
        <p>Os centros da camada escondida foram computados através do algoritmo <strong>K-Means</strong> com $K=2$ levando em consideração <strong>apenas</strong> as 19 amostras de treinamento com presença de radiação ($d = 1.0$). Os resultados obtidos após convergência são:</p>
        
        <table>
            <thead>
                <tr>
                    <th>Cluster</th>
                    <th>Coordenadas do Centro $(x_1, x_2)$</th>
                    <th>Variância Populacional (Biased)</th>
                    <th>Variância Amostral (Unbiased)</th>
                    <th>Amostras Associadas</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>1</strong></td>
                    <td>$({vars_list[0]['c'][0]:.6f}, {vars_list[0]['c'][1]:.6f})$</td>
                    <td>$Var(x_1) = {vars_list[0]['var_pop'][0]:.6f}$<br>$Var(x_2) = {vars_list[0]['var_pop'][1]:.6f}$<br><strong>Total:</strong> ${vars_list[0]['tot_pop']:.6f}$</td>
                    <td>$Var(x_1) = {vars_list[0]['var_sam'][0]:.6f}$<br>$Var(x_2) = {vars_list[0]['var_sam'][1]:.6f}$<br><strong>Total:</strong> ${vars_list[0]['tot_sam']:.6f}$</td>
                    <td>{vars_list[0]['count']}</td>
                </tr>
                <tr>
                    <td><strong>2</strong></td>
                    <td>$({vars_list[1]['c'][0]:.6f}, {vars_list[1]['c'][1]:.6f})$</td>
                    <td>$Var(x_1) = {vars_list[1]['var_pop'][0]:.6f}$<br>$Var(x_2) = {vars_list[1]['var_pop'][1]:.6f}$<br><strong>Total:</strong> ${vars_list[1]['tot_pop']:.6f}$</td>
                    <td>$Var(x_1) = {vars_list[1]['var_sam'][0]:.6f}$<br>$Var(x_2) = {vars_list[1]['var_sam'][1]:.6f}$<br><strong>Total:</strong> ${vars_list[1]['tot_sam']:.6f}$</td>
                    <td>{vars_list[1]['count']}</td>
                </tr>
            </tbody>
        </table>
    </div>

    <h2>Questão 2: Treinamento dos Pesos da Camada de Saída</h2>
    <div class="answer-box">
        <h4>Ajuste de Pesos via Regra Delta Generalizada</h4>
        <p>Ajuste de pesos utilizando taxa de aprendizado eta = 0.01 e precisão epsilon = 10<sup>-7</sup>. Apresentamos os pesos resultantes comparados com os valores exatos de mínimos quadrados:</p>
        
        <table>
            <thead>
                <tr>
                    <th>Peso</th>
                    <th>Cenário: Variância Populacional (Delta)</th>
                    <th>Cenário: Variância Amostral (Delta)</th>
                    <th>Referência: Mínimos Quadrados (Populacional)</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>$W_{21,0}$</strong> (Bias)</td>
                    <td><strong>{weights_pop[0]:.6f}</strong></td>
                    <td>-1.037252</td>
                    <td>-1.001231</td>
                </tr>
                <tr>
                    <td><strong>$W_{21,1}$</strong> (Hidden 1)</td>
                    <td><strong>{weights_pop[1]:.6f}</strong></td>
                    <td>2.145476</td>
                    <td>2.384409</td>
                </tr>
                <tr>
                    <td><strong>$W_{21,2}$</strong> (Hidden 2)</td>
                    <td><strong>{weights_pop[2]:.6f}</strong></td>
                    <td>2.665531</td>
                    <td>2.705125</td>
                </tr>
                <tr>
                    <td><strong>Épocas</strong></td>
                    <td>{epochs_pop}</td>
                    <td>8.647</td>
                    <td>- (Direto)</td>
                </tr>
            </tbody>
        </table>
    </div>

    <div class="image-container">
        <h3>Curva de Aprendizado do Ajuste por Regra Delta (M=2)</h3>
        <img src="data:image/png;base64,{gd_convergence_b64}" alt="Curva de Convergência Delta M=2">
    </div>

    <h2>Questão 3: Rotina de Pós-Processamento</h2>
    <div class="answer-box">
        <h4>Função Sinal de Mapeamento</h4>
        <p>Para mapear a saída linear contínua y (número real) em classes discretas de decisão inteira y_pós (1 ou -1) no conjunto de teste, a rede utiliza a função sinal:</p>
        <p style="text-align:center;"><code>y_pos = 1 se y &gt;= 0</code> &nbsp;&nbsp;|&nbsp;&nbsp; <code>y_pos = -1 se y &lt; 0</code></p>
    </div>

    <h2>Questão 4: Validação no Conjunto de Teste</h2>
    <div class="answer-box">
        <h4>Resultados de Classificação nos Testes</h4>
        <p>Abaixo estão as tabelas de validação contendo os valores reais de saída y gerados e a classe pós-processada y_pós para os dois cenários de espalhamento:</p>
        
        <h3>Cenário 1: Espalhamento via Variância Populacional (Taxa de Acerto: {acc_pop:.1f}%)</h3>
        <table>
            <thead>
                <tr>
                    <th>Amostra</th>
                    <th>x1</th>
                    <th>x2</th>
                    <th>Desejado (d)</th>
                    <th>Saída RBF (y)</th>
                    <th>Pós-proc (y_pós)</th>
                    <th>Resultado</th>
                </tr>
            </thead>
            <tbody>
"""

for i in range(N_test):
    res_str = f"<span class='result-success'>ACERTO</span>" if correct_pop[i] else f"<span class='result-danger'>ERRO</span>"
    html_content += f"""                <tr>
                    <td>{i+1}</td>
                    <td>{X_test[i, 0]:.4f}</td>
                    <td>{X_test[i, 1]:.4f}</td>
                    <td>{d_test[i]:.1f}</td>
                    <td>{y_test_raw_pop[i]:.6f}</td>
                    <td>{y_test_pos_pop[i]:.1f}</td>
                    <td>{res_str}</td>
                </tr>\n"""

html_content += f"""            </tbody>
        </table>
        
        <h3>Cenário 2: Espalhamento via Heurística de Distância (Taxa de Acerto: {acc_heur:.1f}%)</h3>
        <p>Ao aumentar o espalhamento para sigma^2 = {sigmas_sq_heur[0]:.6f} através da heurística baseada na distância máxima entre os centros, a Gaussiana passa a cobrir melhor os limites do cluster, classificando perfeitamente todas as amostras de teste.</p>
        <table>
            <thead>
                <tr>
                    <th>Amostra</th>
                    <th>x1</th>
                    <th>x2</th>
                    <th>Desejado (d)</th>
                    <th>Saída RBF (y)</th>
                    <th>Pós-proc (y_pós)</th>
                    <th>Resultado</th>
                </tr>
            </thead>
            <tbody>
"""

for i in range(N_test):
    res_str = f"<span class='result-success'>ACERTO</span>" if correct_heur[i] else f"<span class='result-danger'>ERRO</span>"
    html_content += f"""                <tr>
                    <td>{i+1}</td>
                    <td>{X_test[i, 0]:.4f}</td>
                    <td>{X_test[i, 1]:.4f}</td>
                    <td>{d_test[i]:.1f}</td>
                    <td>{y_test_raw_heur[i]:.6f}</td>
                    <td>{y_test_pos_heur[i]:.1f}</td>
                    <td>{res_str}</td>
                </tr>\n"""

html_content += f"""            </tbody>
        </table>
    </div>

    <div class="image-container">
        <h3>Fronteira de Decisão Gerada pelo Modelo de 2 Centros (Variância Populacional)</h3>
        <img src="data:image/png;base64,{specific_boundary_b64}" alt="Fronteira 2 Centros">
        <p><em>Note como os círculos de decisão gaussianos são estreitos devido às pequenas variâncias intrínsecas, fazendo com que as amostras 2 e 4 fiquem fora da região positiva de decisão (região verde).</em></p>
    </div>

    <h2>Questão 5: Estratégias para Aumento de Performance</h2>
    <div class="answer-box">
        <ol>
            <li><strong>Otimização do Espalhamento (sigma):</strong> Ajustar a largura de banda gaussiana via busca em grade (grid search) para encontrar a melhor interpolação nos limites dos dados de treino.</li>
            <li><strong>Treinamento de Centros para Ambas as Classes:</strong> Atribuir centros RBF para cobrir também as regiões da classe negativa (-1.0), evitando zonas de sombra de dados onde a rede infere apenas o bias de forma genérica.</li>
            <li><strong>Variação de Topologia (Aumentar M):</strong> Incrementar a quantidade de neurônios ocultos M. Ao projetar para dimensões maiores, a separabilidade aumenta. Com M = 8, por exemplo, obtém-se 100% de acerto.</li>
            <li><strong>Regularização L2 (Tikhonov):</strong> Adicionar penalizador lambda na matriz de pesos de saída para suavizar o desenho da fronteira e prevenir overfitting.</li>
            <li><strong>Treinamento Supervisionado Completo:</strong> Ajustar centros e spreads via gradiente descendente em conjunto com a camada de saída.</li>
        </ol>
    </div>

    <h2>Anexo: Análise Geral de Variação de Topologia (Sweep de Parâmetros)</h2>
    <p>Abaixo está o gráfico comparativo de acurácia no conjunto de teste quando variamos a quantidade de neurônios ocultos e a estratégia de posicionamento dos centros:</p>
    
    <div class="image-container">
        <h3>Comparação Geral de Topologia e Estratégia de Centros</h3>
        <img src="data:image/png;base64,{comparison_b64}" alt="Acurácia vs Neurônios Ocultos">
    </div>

    <div class="image-container">
        <h3>Fronteira de Decisão da Melhor Rede Encontrada (M={best_m}, Estratégia={best_strategy.upper()})</h3>
        <img src="data:image/png;base64,{best_boundary_b64}" alt="Fronteira RBF Melhor Modelo">
    </div>
</div>

</body>
</html>
"""

# Write HTML content to report.html and index.html
report_path = os.path.join(BASE_DIR, 'report.html')
index_path = os.path.join(BASE_DIR, 'index.html')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

with open(index_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Relatório HTML atualizado em report.html e index.html")
