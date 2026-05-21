"""
Testes unitários para a classe RBFNetwork.
Execute com: python3 test_rbf.py
"""

import numpy as np
import sys
import os

# Garante que o diretório pai está no path
sys.path.insert(0, os.path.dirname(__file__))

from rbf import RBFNetwork
from load_data import load_data


def test_predict_shape():
    """predict() deve retornar array de shape (N,)."""
    X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))
    model = RBFNetwork(n_centers=5)
    model.train_least_squares(X, d, seed=42)
    preds = model.predict(X)
    assert preds.shape == (40,), f"Esperado (40,), obtido {preds.shape}"
    print("  [OK] test_predict_shape")


def test_predict_values():
    """Todos os valores preditos devem ser -1.0 ou +1.0."""
    X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))
    model = RBFNetwork(n_centers=5)
    model.train_least_squares(X, d, seed=42)
    preds = model.predict(X)
    assert set(preds).issubset({-1.0, 1.0}), f"Valores inesperados: {set(preds)}"
    print("  [OK] test_predict_values")


def test_kmeans_centers_shape():
    """O algoritmo K-Means deve produzir centros de formato (n_centers, n_features)."""
    X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))
    model = RBFNetwork(n_centers=6, center_strategy='kmeans')
    model.train_least_squares(X, d, seed=42)
    assert model.centers.shape == (6, 2), f"Esperado (6, 2), obtido {model.centers.shape}"
    print("  [OK] test_kmeans_centers_shape")


def test_heuristic_spreads_shape():
    """A heurística deve produzir sigmas de formato (n_centers,)."""
    X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))
    model = RBFNetwork(n_centers=8, spread_strategy='heuristic')
    model.train_least_squares(X, d, seed=42)
    assert model.sigmas.shape == (8,), f"Esperado (8,), obtido {model.sigmas.shape}"
    print("  [OK] test_heuristic_spreads_shape")


def test_least_squares_accuracy():
    """O treinamento via mínimos quadrados deve obter acurácia aceitável."""
    X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))
    model = RBFNetwork(n_centers=10, center_strategy='kmeans')
    model.train_least_squares(X, d, seed=42)
    acc = model.accuracy(X, d)
    assert acc >= 0.95, f"Esperado acurácia >= 95%, obtida {acc*100:.1f}%"
    print(f"  [OK] test_least_squares_accuracy — obtida acurácia de {acc*100:.1f}%")


def test_gradient_descent_decreases_mse():
    """O treinamento por gradiente descendente deve reduzir o MSE ao longo do tempo."""
    X, d = load_data(os.path.join(os.path.dirname(__file__), 'training_data.csv'))
    model = RBFNetwork(n_centers=5, center_strategy='kmeans')
    mse_history = model.train_gradient_descent(X, d, eta=0.05, max_epochs=200, seed=42)
    assert len(mse_history) > 0, "Histórico de MSE não deve estar vazio"
    assert mse_history[-1] < mse_history[0], f"MSE final ({mse_history[-1]:.4f}) não é menor que MSE inicial ({mse_history[0]:.4f})"
    print(f"  [OK] test_gradient_descent_decreases_mse — MSE caiu de {mse_history[0]:.4f} para {mse_history[-1]:.4f}")


if __name__ == '__main__':
    print("Executando testes da RBF...\n")
    tests = [
        test_predict_shape,
        test_predict_values,
        test_kmeans_centers_shape,
        test_heuristic_spreads_shape,
        test_least_squares_accuracy,
        test_gradient_descent_decreases_mse,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    if failed == 0:
        print(f"Todos os {passed} testes passaram.")
    else:
        print(f"{passed} passaram, {failed} falharam.")
        sys.exit(1)
