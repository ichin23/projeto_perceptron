import os
import numpy as np
from rbf import RBFNetwork
from load_data import load_data

# Load full dataset (150 samples)
X, d = load_data()
# For classification we consider presence of radiation when d >= np.mean(d)
threshold = np.mean(d)
mask_presence = d >= threshold
X_pos = X[mask_presence]
d_pos = d[mask_presence]

# 1. K-means on positive samples, 2 clusters
k = 2
# Simple K-means (use same implementation as in RBFNetwork)
model_k = RBFNetwork(n_centers=k, center_strategy='kmeans', spread_strategy='constant', spread_val=1.0)
# Force selection of centers using positive samples only
model_k._select_centers(X_pos, seed=42)
centers = model_k.centers
# Compute variance (average squared distance to center) for each cluster
variances = []
for i in range(k):
    pts = X_pos[np.argmin(np.linalg.norm(X_pos[:, None] - centers[None, :, :], axis=2), axis=1) == i]
    if pts.shape[0] > 0:
        var = np.mean(np.sum((pts - centers[i])**2, axis=1))
    else:
        var = 0.0
    variances.append(var)

# 2. Train output layer with delta rule (LMS) on full training set using the two hidden units
model = RBFNetwork(n_centers=k, center_strategy='random', spread_strategy='heuristic')
# Use same seeds for reproducibility
model.train_gradient_descent(X, d, eta=0.01, epsilon=1e-7, max_epochs=5000, seed=123)
weights = model.weights  # includes bias as first element

# 3. Post‑processing: sign function

def postprocess(y):
    return np.where(y >= 0, 1, -1)

# 4. Validation on the 15 test samples provided in the conversation
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
y_true = np.where(TEST_SAMPLES[:, 3] >= threshold, 1, -1)  # assume same threshold for truth

# Predict raw output
y_raw = model.predict(X_test)
# Apply sign
y_pred = postprocess(y_raw)
accuracy = np.mean(y_pred == y_true) * 100

# 5. Suggestions to improve accuracy
suggestions = [
    "Ajustar o número de neurônios ocultos (centros) – usar mais de 2 centros para capturar melhor a variabilidade dos padrões.",
    "Utilizar uma estratégia de seleção de centros mais robusta, como K‑Means++ ou inicialização baseada em densidade dos dados.",
    "Normalizar/escalar as características (x1, x2, x3) antes do treinamento para evitar que atributos com maior escala dominem o cálculo das distâncias.",
    "Experimentar diferentes funções de base (ex.: funções multiquadráticas ou funções sigmoides) em vez da gaussiana padrão.",
    "Aplicar regularização ao treinamento da camada de saída (por exemplo, ridge regression) para reduzir over‑fitting.",
    "Aumentar o conjunto de treinamento, ou usar validação cruzada para escolher hiper‑parâmetros como a taxa de aprendizado η e o critério de parada ε.",
]

# Output results
out_path = os.path.join(os.path.dirname(__file__), 'answers.txt')
with open(out_path, 'w') as f:
    f.write('1. Centros (k‑means) nos padrões com presença de radiação:\n')
    for i, c in enumerate(centers, 1):
        f.write(f'   Centro {i}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})  Variância: {variances[i-1]:.6f}\n')
    f.write('\n2. Pesos da camada de saída após treinamento delta (bias primeiro):\n')
    f.write('   ' + '  '.join(f'{w:.6f}' for w in weights) + '\n')
    f.write('\n3. Função de pós‑processamento: sinal(y) = 1 se y >= 0, -1 caso contrário.\n')
    f.write('\n4. Taxa de acerto no conjunto de teste: {:.2f}%\n'.format(accuracy))
    f.write('\n5. Estratégias para melhorar a taxa de acerto:\n')
    for s in suggestions:
        f.write('   - ' + s + '\n')

print('Answers written to', out_path)
