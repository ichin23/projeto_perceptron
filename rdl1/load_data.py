import numpy as np


def load_data(filepath='training_data.csv'):
    """
    Load the RBF radiation training or test dataset.

    Reads the CSV file, separates features (x1, x2) from labels (d).

    Parameters
    ----------
    filepath : str
        Path to the CSV file (default: 'training_data.csv')

    Returns
    -------
    X : np.ndarray, shape (N, 2)
        Input matrix of features [x1, x2]
    d : np.ndarray, shape (N,)
        Target labels: -1.0 (absence of radiation) or +1.0 (presence of radiation)
    """
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    X = data[:, :2]  # features: x1, x2
    d = data[:, 2]   # labels: -1 or +1
    return X, d


if __name__ == '__main__':
    X, d = load_data('training_data.csv')
    print(f'X shape: {X.shape}')
    print(f'd shape: {d.shape}')
    print(f'First features:\n{X[:3]}')
    print(f'First labels: {d[:3]}')
