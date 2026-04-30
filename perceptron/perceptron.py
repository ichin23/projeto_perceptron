import numpy as np

class Perceptron:
    """
    Perceptron classifier using the Supervised Hebbian Rule.
    
    Constraints:
    - Bias term x0 = -1.
    - Learning rate eta = 0.01.
    - Activation: Sign function (y = 1 if u >= 0 else -1).
    """
    
    def __init__(self, learning_rate=0.01, initial_weights=None, n_features=3):
        self.eta = learning_rate
        if initial_weights is not None:
            # Expecting n_features + 1 (including bias weight w0)
            self.weights = np.array(initial_weights, dtype=float)
        else:
            # Random weights [0, 1] including bias weight
            self.weights = np.random.rand(n_features + 1)
            
    def predict(self, x):
        """
        Predicts the class for a given input vector x.
        x: array of shape (n_features,)
        """
        # Inclusion of bias term x0 = -1
        # input vector becomes [-1, x1, x2, x3, ...]
        x_with_bias = np.insert(x, 0, -1)
        
        # Weighted sum u = w0*x0 + w1*x1 + ...
        u = np.dot(self.weights, x_with_bias)
        
        # Sign activation function
        y = 1 if u >= 0 else -1
        return y
        
    def train_step(self, x, d):
        """
        Performs a single weight update based on the Supervised Hebbian Rule.
        x: input vector
        d: desired target (-1 or 1)
        Returns: 1 if error occurred, 0 otherwise.
        """
        y = self.predict(x)
        error = d - y
        
        if error != 0:
            # w_new = w_old + eta * (d - y) * x_with_bias
            x_with_bias = np.insert(x, 0, -1)
            self.weights += self.eta * error * x_with_bias
            return 1 # Error occurred
        
        return 0 # No error
