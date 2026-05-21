import numpy as np

class RBFNetwork:
    """RBF Neural Network for regression mapping 3 inputs to a scalar output.

    The network consists of:
    1. Input layer with 3 features (x1, x2, x3).
    2. Hidden layer with N radial basis functions (Gaussian).
    3. Linear output layer with bias term.
    """

    def __init__(self, n_centers=5, center_strategy='random', spread_strategy='heuristic', spread_val=1.0):
        self.n_centers = n_centers
        self.center_strategy = center_strategy
        self.spread_strategy = spread_strategy
        self.spread_val = spread_val
        self.centers = None
        self.sigmas = None
        self.weights = None  # includes bias at index 0

    # ---------------------------------------------------------------------
    # Center selection
    # ---------------------------------------------------------------------
    def _select_centers(self, X, seed=None):
        """Select RBF centers.

        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Input data.
        seed : int or None
            Random seed for reproducibility when using the 'random' strategy.
        """
        if self.center_strategy == 'random':
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], self.n_centers, replace=False)
            self.centers = X[idx].copy()
        elif self.center_strategy == 'kmeans':
            # Simple K-Means implementation (reuse from previous project)
            self.centers = self._kmeans(X, self.n_centers, seed=seed)
        else:
            raise ValueError(f"Unknown center strategy: {self.center_strategy}")

    def _kmeans(self, X, k, max_iters=100, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], k, replace=False)
        centroids = X[idx].copy()
        for _ in range(max_iters):
            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                pts = X[labels == i]
                if len(pts) > 0:
                    new_centroids[i] = pts.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            if np.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids
        return centroids

    # ---------------------------------------------------------------------
    # Spread (sigma) calculation
    # ---------------------------------------------------------------------
    def _calculate_spreads(self):
        if self.spread_strategy == 'constant':
            self.sigmas = np.full(self.n_centers, self.spread_val)
        elif self.spread_strategy == 'heuristic':
            # Heuristic based on max inter‑center distance
            dists = []
            for i in range(self.n_centers):
                for j in range(i + 1, self.n_centers):
                    dists.append(np.linalg.norm(self.centers[i] - self.centers[j]))
            d_max = max(dists) if dists else 1.0
            sigma = d_max / np.sqrt(2 * self.n_centers)
            sigma = max(sigma, 1e-4)
            self.sigmas = np.full(self.n_centers, sigma)
        else:
            raise ValueError(f"Unknown spread strategy: {self.spread_strategy}")

    # ---------------------------------------------------------------------
    # Gaussian activation matrix Φ
    # ---------------------------------------------------------------------
    def _gaussian_rbf(self, X):
        N = X.shape[0]
        Phi = np.empty((N, self.n_centers))
        for j in range(self.n_centers):
            diff = X - self.centers[j]
            dist_sq = np.sum(diff ** 2, axis=1)
            Phi[:, j] = np.exp(-dist_sq / (2 * (self.sigmas[j] ** 2)))
        return Phi

    # ---------------------------------------------------------------------
    # Training via Gradient Descent (LMS)
    # ---------------------------------------------------------------------
    def train_gradient_descent(self, X, d, eta=0.01, epsilon=1e-7, max_epochs=5000, seed=None):
        """Train output weights using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (N, 3)
            Input matrix.
        d : np.ndarray, shape (N,)
            Target values.
        eta : float
            Learning rate.
        epsilon : float
            Stopping criterion on change of MSE.
        max_epochs : int
            Upper bound on epochs.
        seed : int or None
            Random seed – used for both center selection and initial weight
            initialization. Different seeds guarantee distinct initial weights.
        """
        # Select centres and spreads (deterministic for a given seed)
        self._select_centers(X, seed=seed)
        self._calculate_spreads()
        # Compute hidden activation matrix
        Phi = self._gaussian_rbf(X)
        # Add bias column
        Phi_bias = np.hstack([np.ones((X.shape[0], 1)), Phi])
        # Initialise output weights uniformly in [0, 1]
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0.0, 1.0, self.n_centers + 1)
        mse_history = []
        prev_mse = None
        for epoch in range(max_epochs):
            # Linear output
            y_raw = Phi_bias @ self.weights
            errors = d - y_raw
            # LMS weight update (batch version)
            self.weights += eta * (Phi_bias.T @ errors) / X.shape[0]
            mse = np.mean(errors ** 2)
            mse_history.append(mse)
            if prev_mse is not None and abs(prev_mse - mse) < epsilon:
                break
            prev_mse = mse
        return mse_history

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def predict(self, X):
        Phi = self._gaussian_rbf(X)
        Phi_bias = np.hstack([np.ones((X.shape[0], 1)), Phi])
        return Phi_bias @ self.weights

    # ---------------------------------------------------------------------
    # Utility – MSE
    # ---------------------------------------------------------------------
    def mse(self, X, d):
        preds = self.predict(X)
        return np.mean((d - preds) ** 2)

    # ---------------------------------------------------------------------
    # Utility – Accuracy for classification tasks (not used here)
    # ---------------------------------------------------------------------
    def accuracy(self, X, d):
        preds = np.sign(self.predict(X))
        return np.mean(preds == d)

# End of RBFNetwork class
