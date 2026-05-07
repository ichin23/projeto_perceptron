"""
ADALINE — Adaptive Linear Neuron
Regra de aprendizado: Widrow-Hoff (LMS / Delta Rule)

Classificação de sinais industriais ruidosos para roteamento de válvulas:
  d = -1 → Válvula A
  d = +1 → Válvula B
"""

import numpy as np


class ADALINE:
    """
    Implementação da rede ADALINE com regra de Widrow-Hoff (online).

    Parâmetros
    ----------
    eta : float
        Taxa de aprendizado (default: 0.0025)
    epsilon : float
        Precisão para critério de parada: |MSE(t) - MSE(t-1)| < epsilon
        (default: 1e-6)
    """

    def __init__(self, eta: float = 0.0025, epsilon: float = 1e-6):
        self.eta = eta
        self.epsilon = epsilon
        self.weights: np.ndarray | None = None
        self.mse_history: list[float] = []

    def train(self, X: np.ndarray, d: np.ndarray) -> int:
        """
        Treinamento online pela regra de Widrow-Hoff (Delta Rule).

        Para cada padrão (xi, di):
            net_i = wᵀ · xi          (ativação linear)
            error_i = di - net_i      (erro)
            Δw = η · error_i · xi    (atualização)
            w = w + Δw

        Parada: |MSE(época t) - MSE(época t-1)| < ε

        Parâmetros
        ----------
        X : np.ndarray, shape (N, n)
            Matriz de entradas COM bias prepended (x0=1)
        d : np.ndarray, shape (N,)
            Rótulos alvo: -1 ou +1

        Retorna
        -------
        int
            Número de épocas até convergência
        """
        N, n = X.shape
        # Pesos inicializados externamente via set_weights() ou aleatoriamente aqui
        if self.weights is None:
            self.weights = np.random.uniform(0, 1, n)

        self.mse_history = []
        prev_mse = None

        for epoch in range(1, 100_001):
            errors = np.zeros(N)

            for i in range(N):
                net = np.dot(self.weights, X[i])   # ativação linear
                error = d[i] - net                  # erro delta
                self.weights = self.weights + self.eta * error * X[i]
                errors[i] = error

            mse = float(np.mean(errors ** 2))
            self.mse_history.append(mse)

            if prev_mse is not None and abs(mse - prev_mse) < self.epsilon:
                return epoch
            prev_mse = mse

        return 100_000  # atingiu limite máximo de épocas

    def set_weights(self, weights: np.ndarray) -> None:
        """Define os pesos iniciais externamente (para reproducibilidade)."""
        self.weights = weights.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Classifica entradas usando a função sinal (sign) da saída linear.

        Retorna array de -1.0 ou +1.0.
        """
        net = X @ self.weights
        return np.sign(net)

    def net_output(self, X: np.ndarray) -> np.ndarray:
        """Saída linear bruta (antes da função sinal)."""
        return X @ self.weights

    def accuracy(self, X: np.ndarray, d: np.ndarray) -> float:
        """Acurácia de classificação sobre o conjunto fornecido."""
        predictions = self.predict(X)
        return float(np.mean(predictions == d))
