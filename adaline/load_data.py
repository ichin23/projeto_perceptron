import numpy as np


def load_data(filepath='training_data.csv'):
    """
    Load the ADALINE valve signal training dataset.

    Reads the CSV file, separates features (x1-x4) from labels (d),
    and prepends a bias column (x0 = 1) to the input matrix.

    Parameters
    ----------
    filepath : str
        Path to training_data.csv (default: 'training_data.csv')

    Returns
    -------
    X_bias : np.ndarray, shape (35, 5)
        Input matrix with bias prepended: [x0=1, x1, x2, x3, x4]
    d : np.ndarray, shape (35,)
        Target labels: -1.0 (Valve A) or +1.0 (Valve B)
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    X = data[:, :4]               # features: x1, x2, x3, x4
    d = data[:, 4]                 # labels: -1 or +1
    bias = np.ones((X.shape[0], 1))
    X_bias = np.hstack([bias, X])  # prepend x0=1 → shape (N, 5)
    return X_bias, d


if __name__ == '__main__':
    X, d = load_data()
    print(f'X_bias shape: {X.shape}')          # (35, 5)
    print(f'd shape:      {d.shape}')           # (35,)
    print(f'Bias column (x0): {X[:3, 0]}')     # all 1.0
    print(f'First labels: {d[:5]}')
    print(f'Class A (Valve A, d=-1): {int(sum(d == -1))} patterns')
    print(f'Class B (Valve B, d=+1): {int(sum(d == +1))} patterns')
