import numpy as np


class RBFNetwork:
    """
    Radial Basis Function (RBF) Neural Network.
    
    A 3-layer neural network comprising:
    1. Input layer: 2 inputs (x1, x2)
    2. Hidden layer: Radial Basis Functions (Gaussian)
    3. Output layer: Linear output with bias term, mapped to {-1, +1}
    """
    def __init__(self, n_centers=5, center_strategy='random', spread_strategy='heuristic', spread_val=1.0):
        """
        Parameters
        ----------
        n_centers : int
            Number of hidden layer neurons (RBF centers).
        center_strategy : str
            Strategy to select centers: 'random' or 'kmeans'.
        spread_strategy : str
            Strategy to calculate sigmas: 'constant' or 'heuristic'.
        spread_val : float
            Standard deviation value if spread_strategy is 'constant'.
        """
        self.n_centers = n_centers
        self.center_strategy = center_strategy
        self.spread_strategy = spread_strategy
        self.spread_val = spread_val
        self.centers = None
        self.sigmas = None
        self.weights = None  # includes bias at index 0

    def _kmeans(self, X, k, max_iters=100, seed=42):
        """Simple K-Means clustering algorithm implementation from scratch."""
        np.random.seed(seed)
        idx = np.random.choice(X.shape[0], k, replace=False)
        centroids = X[idx].copy()
        
        for _ in range(max_iters):
            # Calculate distance of each point to all centroids
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                points = X[labels == i]
                if len(points) > 0:
                    new_centroids[i] = points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check convergence
            if np.allclose(centroids, new_centroids, atol=1e-5):
                break
            centroids = new_centroids
        return centroids

    def _select_centers(self, X, seed=42):
        """Select RBF centers based on the chosen strategy."""
        if self.center_strategy == 'random':
            np.random.seed(seed)
            idx = np.random.choice(X.shape[0], self.n_centers, replace=False)
            self.centers = X[idx].copy()
        elif self.center_strategy == 'kmeans':
            self.centers = self._kmeans(X, self.n_centers, seed=seed)
        else:
            raise ValueError(f"Unknown center strategy: {self.center_strategy}")

    def _calculate_spreads(self):
        """Calculate sigmas (standard deviation / spread parameter) for each center."""
        if self.spread_strategy == 'constant':
            self.sigmas = np.full(self.n_centers, self.spread_val)
        elif self.spread_strategy == 'heuristic':
            # Heuristic: sigma = d_max / sqrt(2 * M)
            # where d_max is the maximum distance between any two centers
            distances = []
            for i in range(self.n_centers):
                for j in range(i + 1, self.n_centers):
                    dist = np.linalg.norm(self.centers[i] - self.centers[j])
                    distances.append(dist)
            d_max = max(distances) if len(distances) > 0 else 1.0
            sigma = d_max / np.sqrt(2 * self.n_centers)
            # Avoid too small sigmas
            if sigma < 1e-4:
                sigma = 1.0
            self.sigmas = np.full(self.n_centers, sigma)
        else:
            raise ValueError(f"Unknown spread strategy: {self.spread_strategy}")

    def _gaussian_rbf(self, X):
        """Compute the RBF hidden layer activation matrix Phi."""
        N = X.shape[0]
        Phi = np.zeros((N, self.n_centers))
        for j in range(self.n_centers):
            diff = X - self.centers[j]
            dist_sq = np.sum(diff ** 2, axis=1)
            Phi[:, j] = np.exp(-dist_sq / (2 * (self.sigmas[j] ** 2)))
        return Phi

    def train_least_squares(self, X, d, seed=42):
        """
        Train the output weights using the exact Least Squares method (pseudo-inverse).
        """
        self._select_centers(X, seed=seed)
        self._calculate_spreads()
        
        Phi = self._gaussian_rbf(X)
        # Prepend bias column of ones
        Phi_bias = np.hstack([np.ones((X.shape[0], 1)), Phi])
        
        # Closed-form solution: w = pinv(Phi) @ d
        self.weights = np.linalg.pinv(Phi_bias) @ d

    def train_gradient_descent(self, X, d, eta=0.01, epsilon=1e-5, max_epochs=5000, seed=42):
        """
        Train the output weights using gradient descent (LMS/Delta rule).
        """
        self._select_centers(X, seed=seed)
        self._calculate_spreads()
        
        Phi = self._gaussian_rbf(X)
        Phi_bias = np.hstack([np.ones((X.shape[0], 1)), Phi])
        
        # Initialize weights randomly in [-0.5, 0.5]
        np.random.seed(seed)
        self.weights = np.random.uniform(-0.5, 0.5, self.n_centers + 1)
        
        mse_history = []
        
        for epoch in range(max_epochs):
            # Calculate raw outputs
            y_raw = Phi_bias @ self.weights
            errors = d - y_raw
            
            # Update weights: w = w + eta * error * Phi
            self.weights += eta * (Phi_bias.T @ errors) / X.shape[0]
            
            # Compute Mean Squared Error (MSE)
            mse = np.mean(errors ** 2)
            mse_history.append(mse)
            
            # Check stopping criterion: change in MSE
            if epoch > 0 and abs(mse_history[-1] - mse_history[-2]) < epsilon:
                break
                
        return mse_history

    def predict(self, X):
        """Predict the class labels {-1, 1} for the input X."""
        Phi = self._gaussian_rbf(X)
        Phi_bias = np.hstack([np.ones((X.shape[0], 1)), Phi])
        y_raw = Phi_bias @ self.weights
        return np.where(y_raw >= 0, 1.0, -1.0)

    def accuracy(self, X, d):
        """Calculate the classification accuracy of the network."""
        predictions = self.predict(X)
        return np.mean(predictions == d)
