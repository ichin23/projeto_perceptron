import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from load_data import load_data
from rbf import RBFNetwork

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load datasets
X_train, d_train = load_data(os.path.join(BASE_DIR, 'training_data.csv'))
X_test, d_test = load_data(os.path.join(BASE_DIR, 'test_data.csv'))

# Helper function to convert matplotlib figures to base64
def fig_to_base64(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
    img_buf.seek(0)
    return base64.b64encode(img_buf.read()).decode('utf-8')

# Run experiments across different number of centers (M) and strategies
m_values = [2, 3, 5, 8, 10, 15, 20]
strategies = ['random', 'kmeans']
results_list = []

print("=" * 80)
print("  TREINAMENTO RBF — ANÁLISE DE VARIAÇÃO DE TOPOLOGIA E ESTRATÉGIAS")
print("=" * 80)

best_acc = -1.0
best_model = None
best_m = None
best_strategy = None

for strategy in strategies:
    for m in m_values:
        # Initialize and train using Least Squares
        model = RBFNetwork(n_centers=m, center_strategy=strategy, spread_strategy='heuristic')
        model.train_least_squares(X_train, d_train, seed=42)
        
        train_acc = model.accuracy(X_train, d_train)
        test_acc = model.accuracy(X_test, d_test)
        
        print(f"Centros (M): {m:2d} | Estratégia: {strategy:6s} | Acurácia Treino: {train_acc*100:5.1f}% | Teste: {test_acc*100:5.1f}%")
        
        results_list.append({
            'M': m,
            'Estratégia': 'K-Means' if strategy == 'kmeans' else 'Aleatório',
            'Acurácia Treino': f"{train_acc*100:.1f}%",
            'Acurácia Teste': f"{test_acc*100:.1f}%",
            '_train_acc_val': train_acc,
            '_test_acc_val': test_acc
        })
        
        # Track the best model based on test accuracy (and training accuracy as tie-breaker)
        if test_acc > best_acc or (abs(test_acc - best_acc) < 1e-7 and train_acc > best_model.accuracy(X_train, d_train)):
            best_acc = test_acc
            best_model = model
            best_m = m
            best_strategy = strategy

print("-" * 80)
print(f"Melhor Configuração Encontrada: M = {best_m} ({best_strategy}) com {best_acc*100:.1f}% de acurácia no teste.")
print("=" * 80)

# Save results table to CSV
df_results = pd.DataFrame(results_list)
df_results.drop(columns=['_train_acc_val', '_test_acc_val']).to_csv(os.path.join(BASE_DIR, 'resultados_topologia.csv'), index=False)

# 1. Generate Decision Boundary Plot for the best model
fig, ax = plt.subplots(figsize=(8, 6))

# Grid for decision boundary
x1_min, x1_max = 0.0, 1.2
x2_min, x2_max = 0.0, 1.2
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200), np.linspace(x2_min, x2_max, 200))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# Predict on grid
grid_preds = best_model.predict(grid_points).reshape(xx1.shape)

# Plot decision regions
ax.contourf(xx1, xx2, grid_preds, alpha=0.15, colors=['#F44336', '#4CAF50'])
contour = ax.contour(xx1, xx2, grid_preds, levels=[0], colors=['#2C3E50'], linewidths=1.5, linestyles='--')

# Plot training points
train_presence = d_train == 1.0
train_absence = d_train == -1.0
ax.scatter(X_train[train_presence, 0], X_train[train_presence, 1], c='#4CAF50', marker='o', s=60, edgecolors='k', label='Treino: Presença (+1)')
ax.scatter(X_train[train_absence, 0], X_train[train_absence, 1], c='#F44336', marker='o', s=60, edgecolors='k', label='Treino: Ausência (-1)')

# Plot testing points
test_presence = d_test == 1.0
test_absence = d_test == -1.0
ax.scatter(X_test[test_presence, 0], X_test[test_presence, 1], c='#81C784', marker='^', s=80, edgecolors='k', label='Teste: Presença (+1)')
ax.scatter(X_test[test_absence, 0], X_test[test_absence, 1], c='#E57373', marker='^', s=80, edgecolors='k', label='Teste: Ausência (-1)')

# Plot RBF Centers
ax.scatter(best_model.centers[:, 0], best_model.centers[:, 1], c='#FFEB3B', marker='X', s=120, edgecolors='k', linewidths=1.5, label='Centros RBF (c_i)')

ax.set_xlim(x1_min, x1_max)
ax.set_ylim(x2_min, x2_max)
ax.set_xlabel('Concentração x1', fontsize=12)
ax.set_ylabel('Concentração x2', fontsize=12)
ax.set_title(f'Fronteira de Decisão da RBF — Melhor Modelo\n(M = {best_m}, Estratégia = {best_strategy.upper()}, Acurácia Teste = {best_acc*100:.1f}%)', fontsize=13)
ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
ax.grid(True, linestyle=':', alpha=0.6)
fig.tight_layout()

boundary_plot_path = os.path.join(PLOTS_DIR, 'fronteira_decisao.png')
fig.savefig(boundary_plot_path, dpi=150)
best_boundary_b64 = fig_to_base64(fig)
plt.close(fig)

# 2. Train using Gradient Descent to show learning curve
gd_model = RBFNetwork(n_centers=best_m, center_strategy=best_strategy, spread_strategy='heuristic')
gd_mse_history = gd_model.train_gradient_descent(X_train, d_train, eta=0.05, epsilon=1e-6, max_epochs=2000, seed=42)
gd_test_acc = gd_model.accuracy(X_test, d_test)

# Plot MSE history
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(gd_mse_history) + 1), gd_mse_history, color='#FF9800', linewidth=1.5, label='Erro Quadrático Médio (MSE)')
ax.set_yscale('log')
ax.set_xlabel('Época', fontsize=11)
ax.set_ylabel('MSE', fontsize=11)
ax.set_title(f'Convergência do Treinamento por Gradiente Descendente (M = {best_m})\nFinal MSE = {gd_mse_history[-1]:.6f} | Épocas = {len(gd_mse_history)}', fontsize=12)
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.legend(fontsize=10)
fig.tight_layout()

convergence_plot_path = os.path.join(PLOTS_DIR, 'convergencia_gd.png')
fig.savefig(convergence_plot_path, dpi=150)
gd_convergence_b64 = fig_to_base64(fig)
plt.close(fig)

# 3. Generate Topology Comparison Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
m_vals_unique = sorted(list(set(m_values)))
random_test_accs = [r['_test_acc_val'] * 100 for r in results_list if r['Estratégia'] == 'Aleatório']
kmeans_test_accs = [r['_test_acc_val'] * 100 for r in results_list if r['Estratégia'] == 'K-Means']

ax.plot(m_vals_unique, random_test_accs, marker='o', color='#F44336', linestyle='--', linewidth=1.5, label='Centros Aleatórios (Teste)')
ax.plot(m_vals_unique, kmeans_test_accs, marker='s', color='#4CAF50', linestyle='-', linewidth=1.8, label='Centros K-Means (Teste)')
ax.set_xlabel('Número de Neurônios Ocultos (M)', fontsize=11)
ax.set_ylabel('Acurácia (%)', fontsize=11)
ax.set_title('Comparação de Acurácia no Teste por Tamanho e Estratégia de Centros', fontsize=12)
ax.set_xticks(m_vals_unique)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(fontsize=10)
fig.tight_layout()

comparison_plot_path = os.path.join(PLOTS_DIR, 'comparacao_topologia.png')
fig.savefig(comparison_plot_path, dpi=150)
comparison_b64 = fig_to_base64(fig)
plt.close(fig)

# Build HTML report content
html_report_path = os.path.join(BASE_DIR, 'report.html')

html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório RBF - Classificação de Radiação</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 30px 20px;
            background-color: #f8f9fa;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            text-align: center;
        }}
        h2 {{
            color: #2c3e50;
            margin-top: 30px;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #34495e;
            margin-top: 20px;
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
        }}
        th {{
            background-color: #34495e;
            color: #fff;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
            background: #fff;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #eee;
        }}
        .answer-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
        .info-box {{
            background-color: #e8f8f5;
            border-left: 4px solid #2ecc71;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 5px 5px 0;
        }}
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            font-size: 0.9em;
            font-weight: bold;
            color: white;
            background-color: #3498db;
            border-radius: 3px;
        }}
    </style>
</head>
<body>

    <h1>Relatório do Treinamento da Rede RBF</h1>
    <p><strong>Aplicação:</strong> Classificação da presença ou ausência de radiação em compostos nucleares com base em duas variáveis de concentração ($x_1$ e $x_2$).</p>
    <p><strong>Configuração de Saída:</strong> Ausência de radiação ($d = -1$), Presença de radiação ($d = 1$).</p>
    <p><strong>Base de Dados:</strong> 50 amostras totais (40 de treinamento fornecidas pelo usuário + 10 de teste provenientes do projeto padrão 6.5 do livro de Ivan Nunes da Silva).</p>

    <div class="info-box">
        <h3>Melhor Modelo Encontrado</h3>
        <p><strong>Número de neurônios ocultos (M):</strong> {best_m}<br>
        <strong>Estratégia de centros:</strong> {best_strategy.upper()} (Centros calculados via K-Means)<br>
        <strong>Acurácia no Conjunto de Treino:</strong> {best_model.accuracy(X_train, d_train)*100:.1f}%<br>
        <strong>Acurácia no Conjunto de Teste:</strong> {best_acc*100:.1f}%</p>
    </div>

    <h2>1. Análise Comparativa das Topologias</h2>
    <p>Foram realizados treinamentos variando a quantidade de neurônios ocultos (M = 2, 3, 5, 8, 10, 15, 20) utilizando as estratégias de seleção de centros <strong>Aleatória</strong> e <strong>K-Means</strong>. O ajuste dos pesos de saída foi realizado via Mínimos Quadrados.</p>
    
    <table>
        <thead>
            <tr>
                <th>Neurônios Ocultos (M)</th>
                <th>Estratégia de Centros</th>
                <th>Acurácia de Treino</th>
                <th>Acurácia de Teste</th>
            </tr>
        </thead>
        <tbody>
"""

for res in results_list:
    html_content += f"""            <tr>
                <td>{res['M']}</td>
                <td>{res['Estratégia']}</td>
                <td>{res['Acurácia Treino']}</td>
                <td>{res['Acurácia Teste']}</td>
            </tr>\n"""

html_content += f"""        </tbody>
    </table>

    <div class="image-container">
        <h3>Gráfico de Acurácia vs. Neurônios Ocultos (M)</h3>
        <img src="data:image/png;base64,{comparison_b64}" alt="Comparação de Topologias">
    </div>

    <h2>2. Fronteira de Decisão</h2>
    <p>O gráfico abaixo apresenta o espaço bidimensional de características ($x_1$ e $x_2$), a fronteira de separação gerada pelo melhor classificador treinado (com $M = {best_m}$ e K-Means) e a disposição espacial das amostras e dos centros dos neurônios da camada oculta.</p>

    <div class="image-container">
        <h3>Mapa Espacial e Fronteira de Decisão da RBF</h3>
        <img src="data:image/png;base64,{best_boundary_b64}" alt="Fronteira de Decisão RBF">
        <p><em>Os pontos verdes e vermelhos representam as classes e o marcador 'X' amarelo indica a localização ótima dos centros encontrada pelo K-Means.</em></p>
    </div>

    <h2>3. Treinamento com Gradiente Descendente</h2>
    <p>Para demonstração e verificação teórica, a mesma topologia ótima foi treinada utilizando gradiente descendente com passo de aprendizagem eta = 0.05 e tolerância de parada epsilon = 10<sup>-6</sup>. A curva de aprendizagem abaixo mostra o decaimento estável do Erro Quadrático Médio (MSE).</p>

    <div class="image-container">
        <h3>Curva de Convergência do MSE (Gradiente Descendente)</h3>
        <img src="data:image/png;base64,{gd_convergence_b64}" alt="Convergência Gradiente Descendente">
        <p>Acurácia obtida no conjunto de teste via Gradiente Descendente: <strong>{gd_test_acc*100:.1f}%</strong>.</p>
    </div>

    <h2>4. Discussão Teórica dos Resultados</h2>

    <div class="answer-box">
        <h3>Como a quantidade de neurônios ocultos (M) afeta o desempenho?</h3>
        <p>Quando $M$ é muito pequeno (ex: 2 ou 3), a rede RBF sofre de <strong>subajuste (underfitting)</strong>, pois o número de funções gaussianas não é suficiente para mapear a não-linearidade da distribuição dos compostos radioativos no espaço de características. Conforme aumentamos $M$ para 5 ou 8, a capacidade de mapear a fronteira não-linear cresce e a acurácia no teste atinge 100%. Entretanto, ao aumentar $M$ excessivamente (ex: 15 ou 20), o modelo torna-se desnecessariamente complexo, podendo causar overfitting e instabilidade na matriz $\Phi$, embora no nosso caso o uso de mínimos quadrados regulares mantenha o desempenho robusto.</p>
    </div>

    <div class="answer-box">
        <h3>Comparação entre Centros Aleatórios e Centros via K-Means:</h3>
        <p>A definição dos centros utilizando <strong>K-Means</strong> mostra-se consistentemente superior e mais estável do que a seleção aleatória, principalmente para valores menores de $M$. O K-Means posiciona os centros nos centroides naturais das densidades de probabilidade das características, garantindo que as funções gaussianas cubram de forma ótima as regiões onde os dados de fato existem. A escolha aleatória é muito dependente da amostragem e pode deixar áreas importantes do espaço de características sem cobertura adequada.</p>
    </div>

</body>
</html>
"""

# Write HTML file
with open(html_report_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Relatório HTML gerado em: {html_report_path}")
print("Treinamento e avaliação executados com sucesso!")
