"""
Testes unitários para a classe ADALINE.
Execute com: python3 test_adaline.py
"""

import numpy as np
import sys
import os

# Garante que o diretório pai está no path
sys.path.insert(0, os.path.dirname(__file__))

from adaline import ADALINE
from load_data import load_data


def test_predict_shape():
    """predict() deve retornar array de shape (N,)."""
    X, d = load_data()
    model = ADALINE()
    model.weights = np.ones(X.shape[1])
    preds = model.predict(X)
    assert preds.shape == (35,), f"Esperado (35,), obtido {preds.shape}"
    print("  [OK] test_predict_shape")


def test_predict_values():
    """Todos os valores preditos devem ser -1.0 ou +1.0."""
    X, d = load_data()
    np.random.seed(0)
    model = ADALINE()
    model.weights = np.random.uniform(0, 1, X.shape[1])
    preds = model.predict(X)
    assert set(preds).issubset({-1.0, 0.0, 1.0}), \
        f"Valores inesperados: {set(preds)}"
    print("  [OK] test_predict_values")


def test_train_returns_epoch():
    """train() deve retornar um inteiro positivo (número de épocas)."""
    X, d = load_data()
    np.random.seed(42)
    model = ADALINE(eta=0.0025, epsilon=1e-6)
    epochs = model.train(X, d)
    assert isinstance(epochs, int) and epochs > 0, \
        f"Esperado int positivo, obtido {epochs}"
    print(f"  [OK] test_train_returns_epoch — convergiu em {epochs} épocas")


def test_weights_shape():
    """Após treino, weights deve ter shape (5,) para 4 entradas + bias."""
    X, d = load_data()
    np.random.seed(7)
    model = ADALINE()
    model.train(X, d)
    assert model.weights is not None, "weights não deve ser None após treino"
    assert model.weights.shape == (5,), \
        f"Esperado (5,), obtido {model.weights.shape}"
    print("  [OK] test_weights_shape")


def test_convergence_criterion():
    """Após convergência, a diferença entre MSEs consecutivos deve ser < ε."""
    X, d = load_data()
    np.random.seed(99)
    model = ADALINE(eta=0.0025, epsilon=1e-6)
    model.train(X, d)
    assert len(model.mse_history) >= 2, "Histórico de MSE deve ter ao menos 2 entradas"
    last_diff = abs(model.mse_history[-1] - model.mse_history[-2])
    assert last_diff < 1e-6, \
        f"Diferença final de MSE {last_diff:.2e} não é < 1e-6"
    print(f"  [OK] test_convergence_criterion — Δ MSE final = {last_diff:.2e}")


def test_bias_used():
    """w0 (peso do bias) deve ser não-zero após treinamento."""
    X, d = load_data()
    np.random.seed(13)
    model = ADALINE()
    model.train(X, d)
    assert model.weights[0] != 0.0, "w0 (bias weight) não deve ser zero"
    print(f"  [OK] test_bias_used — w0 = {model.weights[0]:.6f}")


def test_mse_history_populated():
    """mse_history deve ser preenchido com floats durante o treino."""
    X, d = load_data()
    np.random.seed(55)
    model = ADALINE()
    model.train(X, d)
    assert len(model.mse_history) > 0, "mse_history não deve estar vazio"
    assert all(isinstance(v, float) for v in model.mse_history), \
        "mse_history deve conter floats"
    print(f"  [OK] test_mse_history_populated — {len(model.mse_history)} épocas registradas")


if __name__ == '__main__':
    print("Executando testes da ADALINE...\n")
    tests = [
        test_predict_shape,
        test_predict_values,
        test_train_returns_epoch,
        test_weights_shape,
        test_convergence_criterion,
        test_bias_used,
        test_mse_history_populated,
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
