import numpy as np
import pandas as pd

from load_data import load_data
from adaline import ADALINE

# Configurações do treinamento (iguais ao projeto)
ETA = 0.0025
EPSILON = 1e-6
SEEDS = [0, 42, 137, 256, 999]

# 1. Carregar dados de treinamento
X_train, d_train = load_data('training_data.csv')

# 2. Carregar dados de teste
test_data = np.loadtxt('test_data.csv', delimiter=',', skiprows=1)
X_test = test_data  # (15, 4)
# Prepend bias
bias = np.ones((X_test.shape[0], 1))
X_test_bias = np.hstack([bias, X_test])  # (15, 5)

# 3. Treinar os 5 modelos e obter as predições de cada um
models = []
predictions = {}

for run, seed in enumerate(SEEDS, 1):
    np.random.seed(seed)
    w_initial = np.random.uniform(0, 1, X_train.shape[1])
    
    model = ADALINE(eta=ETA, epsilon=EPSILON)
    model.set_weights(w_initial)
    model.train(X_train, d_train)
    models.append(model)
    
    # Prever os 15 sinais de teste
    preds = model.predict(X_test_bias)
    predictions[f'y (T{run})'] = preds

# 4. Construir o DataFrame com os resultados do teste
df_test = pd.DataFrame(test_data, columns=['x1', 'x2', 'x3', 'x4'])
df_test.insert(0, 'Amostra', range(1, 16))

for run in range(1, 6):
    col_name = f'y (T{run})'
    df_test[col_name] = predictions[col_name].astype(int)

# 5. Imprimir a tabela e salvar em CSV
print("="*60)
print("  Tabela de Classificação dos Sinais de Teste")
print("="*60)
print(df_test.to_string(index=False))

df_test.to_csv('resultados_teste.csv', index=False)
