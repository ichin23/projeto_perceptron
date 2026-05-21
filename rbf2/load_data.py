import numpy as np
import os


def load_data(filepath=None):
    """
    Load the RBF gasoline injection training dataset.

    Reads the CSV file and separates features (x1, x2, x3) from labels (d).

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. If None, defaults to 'training_data.csv' in the same folder.

    Returns
    -------
    X : np.ndarray, shape (N, 3)
        Input matrix of features [x1, x2, x3]
    d : np.ndarray, shape (N,)
        Target label: gasoline amount to be injected (continuous variable)
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data.csv')
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    X = data[:, 1:4]  # features: x1, x2, x3 (column 0 is the sample index "Amostra")
    d = data[:, 4]   # target gasoline amount: d
    return X, d


if __name__ == '__main__':
    X, d = load_data()
    print(f'X shape: {X.shape}')  # Expected (150, 3)
    print(f'd shape: {d.shape}')  # Expected (150,)
    print(f'First features:\n{X[:3]}')
    print(f'First targets: {d[:3]}')
