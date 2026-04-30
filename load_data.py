import numpy as np
import os

def load_training_data(filepath='training_data.csv'):
    """
    Loads the training data from a CSV file and returns features and targets.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    
    # Load all data as a single matrix
    data = np.genfromtxt(filepath, delimiter=',')
    
    # Check shape: should be (30, 4)
    print(f"Data loaded successfully. Shape: {data.shape}")
    
    # Separate features (x1, x2, x3) and target (d)
    features = data[:, :3]
    targets = data[:, 3]
    
    print(f"Features shape: {features.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return features, targets

if __name__ == "__main__":
    try:
        features, targets = load_training_data()
        
        # Verify specific counts (sanity check)
        expected_patterns = 30
        if features.shape[0] != expected_patterns:
            print(f"ERROR: Expected {expected_patterns} patterns, but found {features.shape[0]}")
            exit(1)
            
        print("Verification PASSED: All 30 patterns loaded correctly.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
