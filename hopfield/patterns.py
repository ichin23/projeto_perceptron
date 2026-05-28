import numpy as np

# Digit patterns encoded as 9x5 grids.
# White pixel is -1, Dark/Grey pixel is +1.

P1 = np.array([
    [-1, -1,  1,  1, -1],
    [-1,  1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1],
    [-1, -1,  1,  1, -1]
])

P2 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1, -1,  1],
    [-1, -1, -1, -1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1, -1, -1, -1],
    [ 1,  1, -1, -1, -1],
    [-1, -1,  1,  1,  1],
    [ 1,  1,  1,  1,  1]
])

P3 = np.array([
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1]
])

P4 = np.array([
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1, -1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1,  1,  1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1],
    [-1, -1, -1,  1,  1]
])

# Export flat vectors for Hopfield network and grid shapes for reconstruction/display
PATTERNS = [P1.flatten(), P2.flatten(), P3.flatten(), P4.flatten()]
NAMES = ["1", "2", "3", "4"]
GRID_SHAPE = (9, 5)

def get_pattern_grid(pattern_vector):
    """Reshapes a flat 45-element pattern vector back to 9x5 grid."""
    return pattern_vector.reshape(GRID_SHAPE)
