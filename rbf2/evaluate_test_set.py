import os
import csv
import numpy as np
from rbf import RBFNetwork
from load_data import load_data

# Test set (first 15 samples from the user's table)
# Columns: x1, x2, x3, d (desired output)
TEST_SAMPLES = np.array([
    [0.5102, 0.7464, 0.0860, 0.5965],
    [0.8401, 0.4490, 0.2719, 0.6790],
    [0.1283, 0.1882, 0.7253, 0.4662],
    [0.2299, 0.1524, 0.7353, 0.5012],
    [0.3209, 0.6229, 0.5233, 0.6810],
    [0.8203, 0.0682, 0.4260, 0.5643],
    [0.3471, 0.8889, 0.1564, 0.5875],
    [0.5762, 0.8292, 0.4116, 0.7853],
    [0.9053, 0.6245, 0.5264, 0.8506],
    [0.8149, 0.0396, 0.6227, 0.6165],
    [0.1016, 0.6382, 0.3173, 0.4957],
    [0.9108, 0.2139, 0.4641, 0.6625],
    [0.2245, 0.0971, 0.6136, 0.4402],
    [0.6423, 0.3229, 0.8567, 0.7663],
    [0.5252, 0.6529, 0.5729, 0.7893],
])

X_test = TEST_SAMPLES[:, :3]
d_test = TEST_SAMPLES[:, 3]

# Load full training dataset
X_train, d_train = load_data()

centers_list = [5, 10, 15]
repetitions = 3

# Prepare folders for results
out_dir = os.path.join(os.path.dirname(__file__), 'evaluation')
os.makedirs(out_dir, exist_ok=True)

# Header for markdown table
md_lines = []
md_lines.append('# Avaliação nos 15 padrões de teste')
md_lines.append('')
md_lines.append('| Amostra | x1 | x2 | x3 | d |')
md_lines.append('|---|---|---|---|---|')
for i, row in enumerate(TEST_SAMPLES, start=1):
    md_lines.append(f'| {i:02d} | {row[0]:.4f} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} |')
md_lines.append('')

# Table for errors per topology/run
md_lines.append('## Erro Relativo Médio (%, Variância %) por Rede e Treinamento')
md_lines.append('')
md_lines.append('| Rede | Treinamento | EQM Médio (%) | Variância (%) |')
md_lines.append('|---|---|---|---|')

# Save raw predictions CSV for possible later use
csv_path = os.path.join(out_dir, 'predictions.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['Rede', 'Treinamento', 'Amostra', 'y_pred']
    writer.writerow(header)
    for n in centers_list:
        for run in range(1, repetitions + 1):
            seed = 1000 * n + run
            model = RBFNetwork(n_centers=n, center_strategy='random', spread_strategy='heuristic')
            model.train_gradient_descent(
                X_train, d_train,
                eta=0.01, epsilon=1e-7, max_epochs=5000, seed=seed,
            )
            y_pred = model.predict(X_test)
            # Record each prediction
            for idx, yp in enumerate(y_pred, start=1):
                writer.writerow([n, f'T{run}', idx, yp])
            # Compute relative errors (%)
            rel_err = np.abs((y_pred - d_test) / d_test) * 100.0
            mean_err = np.mean(rel_err)
            var_err = np.var(rel_err)
            md_lines.append(f'| {n} | T{run} | {mean_err:.4f} | {var_err:.4f} |')

# Write markdown file
md_path = os.path.join(out_dir, 'evaluation.md')
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print('Evaluation completed. Files written to:', out_dir)
