import os
import numpy as np
import matplotlib.pyplot as plt
from rbf import RBFNetwork
from load_data import load_data

# Ensure output directory
out_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(out_dir, exist_ok=True)

# Load full training data
X_train, d_train = load_data()

centers_list = [5, 10, 15]
repetitions = 3
best_histories = {}
best_labels = {}

for n in centers_list:
    best_mse = float('inf')
    best_history = None
    best_run = None
    for run in range(1, repetitions + 1):
        seed = 1000 * n + run
        model = RBFNetwork(n_centers=n, center_strategy='random', spread_strategy='heuristic')
        mse_history = model.train_gradient_descent(
            X_train, d_train,
            eta=0.01,
            epsilon=1e-7,
            max_epochs=5000,
            seed=seed,
        )
        final_mse = mse_history[-1]
        if final_mse < best_mse:
            best_mse = final_mse
            best_history = mse_history
            best_run = run
    best_histories[n] = best_history
    best_labels[n] = f"N={n} (T{best_run})"

# Plot
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True)
for ax, (n, history) in zip(axes, best_histories.items()):
    ax.plot(range(1, len(history) + 1), history, label=best_labels[n])
    ax.set_ylabel('EQM (MSE)')
    ax.set_title(best_labels[n])
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
axes[-1].set_xlabel('Épocas')

plt.tight_layout()
plot_path = os.path.join(out_dir, 'best_mse_curves.png')
plt.savefig(plot_path, dpi=150)
print(f'Plot saved to {plot_path}')
