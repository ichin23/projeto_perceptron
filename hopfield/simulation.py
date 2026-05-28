import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # headless plot saving

from hopfield import HopfieldNetwork
from patterns import PATTERNS, NAMES, GRID_SHAPE

# ─── Configurações ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Fixed seed for reproducibility
np.random.seed(42)

def inject_noise(pattern, noise_level):
    """
    Corrupts a pattern by flipping exactly noise_level * N bits.
    """
    corrupted = pattern.copy()
    n_bits = len(pattern)
    n_flips = int(round(noise_level * n_bits))
    flip_indices = np.array([])
    if n_flips > 0:
        flip_indices = np.random.choice(n_bits, n_flips, replace=False)
        corrupted[flip_indices] *= -1
    return corrupted, flip_indices

def run_simulation():
    print("=" * 70)
    print("  SIMULAÇÃO REDE DE HOPFIELD (45 NEURÔNIOS)")
    print("=" * 70)
    
    # Initialize and train network
    net = HopfieldNetwork(n_neurons=45)
    net.train(PATTERNS)
    
    # ─── Experimento 1: Desempenho a 20% de ruído (9 pixels corrompidos) ──────────
    print("\nExecutando Experimento 1: 20% de ruído...")
    n_trials = 1000
    noise_20 = 0.20 # 9/45 is exactly 20%
    
    exp1_results = []
    sample_runs = {} # To store a sample run of each digit for visualization
    
    for digit_idx, (pattern, name) in enumerate(zip(PATTERNS, NAMES)):
        successes_async = 0
        successes_sync = 0
        sweeps_async_total = 0
        iterations_sync_total = 0
        
        # Save three sample runs for visualization (3 situations of transmission per pattern = 12 total)
        sample_runs[name] = []
        for run_idx in range(3):
            sample_noisy, sample_flips = inject_noise(pattern, noise_20)
            
            # Recover asynchronous
            rec_async, conv_async, hist_async = net.predict_asynchronous(sample_noisy, max_sweeps=100, record_history=True)
            # Recover synchronous
            rec_sync, conv_sync, hist_sync = net.predict_synchronous(sample_noisy, max_iterations=100, record_history=True)
            
            sample_runs[name].append({
                'original': pattern.tolist(),
                'noisy': sample_noisy.tolist(),
                'flips': sample_flips.tolist(),
                'recovered_async': rec_async.tolist(),
                'converged_async': bool(conv_async),
                'success_async': bool(np.array_equal(rec_async, pattern)),
                'history_async_energies': hist_async['energies'],
                'history_async_sweeps': hist_async['sweeps_completed'],
                
                'recovered_sync': rec_sync.tolist(),
                'converged_sync': bool(conv_sync),
                'success_sync': bool(np.array_equal(rec_sync, pattern)),
                'history_sync_energies': hist_sync['energies'],
                'history_sync_iterations': hist_sync['iterations_completed']
            })
        
        # Run 1000 trials
        for trial in range(n_trials):
            noisy, _ = inject_noise(pattern, noise_20)
            
            # Asynchronous
            rec_a, conv_a, hist_a = net.predict_asynchronous(noisy, max_sweeps=100, record_history=False)
            if np.array_equal(rec_a, pattern):
                successes_async += 1
                sweeps_async_total += hist_a['sweeps_completed']
                
            # Synchronous
            rec_s, conv_s, hist_s = net.predict_synchronous(noisy, max_iterations=100, record_history=False)
            if np.array_equal(rec_s, pattern):
                successes_sync += 1
                iterations_sync_total += hist_s['iterations_completed']
                
        rate_async = (successes_async / n_trials) * 100
        rate_sync = (successes_sync / n_trials) * 100
        avg_sweeps_async = (sweeps_async_total / successes_async) if successes_async > 0 else 0
        avg_iters_sync = (iterations_sync_total / successes_sync) if successes_sync > 0 else 0
        
        print(f"Padrão '{name}':")
        print(f"  Amostra Async -> Sucesso: {sample_runs[name][0]['success_async']} em {sample_runs[name][0]['history_async_sweeps']} sweeps. Energia inicial: {sample_runs[name][0]['history_async_energies'][0]:.2f} -> final: {sample_runs[name][0]['history_async_energies'][-1]:.2f}")
        print(f"  Taxa de recuperação (1000 testes):")
        print(f"    Asynchronous: {rate_async:.1f}% (média {avg_sweeps_async:.2f} sweeps)")
        print(f"    Synchronous:  {rate_sync:.1f}% (média {avg_iters_sync:.2f} iterações)")
        
        exp1_results.append({
            'Digito': name,
            'Taxa_Async': rate_async,
            'Média_Sweeps_Async': round(avg_sweeps_async, 2),
            'Taxa_Sync': rate_sync,
            'Média_Iters_Sync': round(avg_iters_sync, 2)
        })
        
    df_exp1 = pd.DataFrame(exp1_results)
    df_exp1.to_csv(os.path.join(BASE_DIR, 'resultados_20_ruido.csv'), index=False)
    
    # ─── Experimento 2: Sensibilidade ao Ruído (0% a 50%) ─────────────────────────
    print("\nExecutando Experimento 2: Análise de sensibilidade de ruído...")
    noise_levels = np.arange(0.0, 0.51, 0.05) # 0%, 5%, 10%, ..., 50%
    n_sensitivity_trials = 500
    
    sensitivity_results = []
    
    for noise in noise_levels:
        row = {'Ruído': f"{int(round(noise * 100))}%"}
        for pattern, name in zip(PATTERNS, NAMES):
            successes_async = 0
            successes_sync = 0
            for _ in range(n_sensitivity_trials):
                noisy, _ = inject_noise(pattern, noise)
                # Async
                rec_a, _, _ = net.predict_asynchronous(noisy, max_sweeps=50, record_history=False)
                if np.array_equal(rec_a, pattern):
                    successes_async += 1
                # Sync
                rec_s, _, _ = net.predict_synchronous(noisy, max_iterations=50, record_history=False)
                if np.array_equal(rec_s, pattern):
                    successes_sync += 1
            
            row[f'Async_D{name}'] = successes_async / n_sensitivity_trials
            row[f'Sync_D{name}'] = successes_sync / n_sensitivity_trials
            
        sensitivity_results.append(row)
        print(f"  Concluído ruído = {int(round(noise * 100))}%")
        
    df_sens = pd.DataFrame(sensitivity_results)
    df_sens.to_csv(os.path.join(BASE_DIR, 'resultados_sensibilidade.csv'), index=False)
    
    # Generate Sensitivity Matplotlib Plot
    plt.figure(figsize=(10, 6))
    colors_async = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    colors_sync = ['#1565C0', '#2E7D32', '#EF6C00', '#6A1B9A']
    
    for i, name in enumerate(NAMES):
        plt.plot(noise_levels * 100, df_sens[f'Async_D{name}'] * 100, 
                 marker='o', linestyle='-', color=colors_async[i], linewidth=2, label=f'Dígito {name} (Async)')
        plt.plot(noise_levels * 100, df_sens[f'Sync_D{name}'] * 100, 
                 marker='x', linestyle='--', color=colors_sync[i], linewidth=1.2, alpha=0.7, label=f'Dígito {name} (Sync)')
                 
    plt.title('Sensibilidade ao Ruído: Taxa de Recuperação vs. Nível de Ruído', fontsize=13)
    plt.xlabel('Porcentagem de Pixels Corrompidos (%)', fontsize=11)
    plt.ylabel('Taxa de Recuperação Fiel (%)', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'taxa_recuperacao.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nGráfico salvo em: {plot_path}")
    
    # Save the sample runs data as JSON for easy consumption by dashboard script
    with open(os.path.join(BASE_DIR, 'sample_runs.json'), 'w', encoding='utf-8') as f:
        json.dump(sample_runs, f, indent=2)
        
    print("\nSimulação concluída e dados salvos com sucesso!")
    print("=" * 70)

if __name__ == '__main__':
    run_simulation()
