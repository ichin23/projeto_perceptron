import os
import csv
import numpy as np
from rbf import RBFNetwork
from load_data import load_data

# Ensure output directory
output_dir = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(output_dir, exist_ok=True)

# Load dataset
X, d = load_data()

# Configurations
centers_list = [5, 10, 15]
repetitions = 3  # three independent trainings per topology

# Prepare summary list
summary = []  # each entry: (topology, run_id, final_mse, epochs)

for n in centers_list:
    for run in range(1, repetitions + 1):
        # Use a distinct seed for each run (e.g., base + run)
        seed = 1000 * n + run
        model = RBFNetwork(n_centers=n, center_strategy='random', spread_strategy='heuristic')
        mse_history = model.train_gradient_descent(
            X, d,
            eta=0.01,
            epsilon=1e-7,
            max_epochs=5000,
            seed=seed,
        )
        final_mse = mse_history[-1]
        epochs = len(mse_history)
        summary.append([n, f"T{run}", final_mse, epochs])
        print(f"Topology N={n}, Run {run}: final MSE={final_mse:.6f}, epochs={epochs}")

# Write CSV
csv_path = os.path.join(output_dir, 'summary.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Topology', 'Training', 'EQM', 'Epocas'])
    for row in summary:
        writer.writerow(row)
print(f"Summary CSV written to {csv_path}")
