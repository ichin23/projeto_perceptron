import numpy as np
import matplotlib.pyplot as plt
from perceptron import Perceptron
import load_data
import os

def generate_report(runs_data, test_results):
    """
    Generates report.html with results, tables, and embedded graphs.
    """
    html_content = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Treinamento Perceptron</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; background-color: #f4f7f6; }}
        h1, h2 {{ color: #2c3e50; text-align: center; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: center; border: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .graph-container {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; }}
        .graph-card {{ background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); width: 400px; }}
        .graph-card img {{ width: 100%; border-radius: 4px; }}
        .qa {{ margin-top: 30px; }}
        .qa h3 {{ color: #e67e22; border-left: 5px solid #e67e22; padding-left: 10px; }}
        .footer {{ text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>Relatório de Treinamento: Perceptron Oil Classifier</h1>
    
    <div class="card">
        <h2>Resultados dos 5 Treinamentos</h2>
        <table>
            <thead>
                <tr>
                    <th rowspan="2">Treinamento</th>
                    <th colspan="4">Vetor de Pesos Inicial</th>
                    <th colspan="4">Vetor de Pesos Final</th>
                    <th rowspan="2">Épocas</th>
                </tr>
                <tr>
                    <th>w0</th><th>w1</th><th>w2</th><th>w3</th>
                    <th>w0</th><th>w1</th><th>w2</th><th>w3</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i, run in enumerate(runs_data):
        wi = run['initial_weights']
        wf = run['final_weights']
        html_content += f"""
                <tr>
                    <td>{i+1}º (T{i+1})</td>
                    <td>{wi[0]:.4f}</td><td>{wi[1]:.4f}</td><td>{wi[2]:.4f}</td><td>{wi[3]:.4f}</td>
                    <td>{wf[0]:.4f}</td><td>{wf[1]:.4f}</td><td>{wf[2]:.4f}</td><td>{wf[3]:.4f}</td>
                    <td>{run['epochs']}</td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Curvas de Aprendizado (Erro por Época)</h2>
        <div class="graph-container">
    """
    
    for i in range(5):
        html_content += f"""
            <div class="graph-card">
                <h3>Treinamento {i+1}</h3>
                <img src="error_run_{i+1}.png" alt="Gráfico Run {i+1}">
            </div>
        """
        
    html_content += """
        </div>
    </div>

    <div class="card">
        <h2>Classificação de Amostras de Teste</h2>
        <table>
            <thead>
                <tr>
                    <th>Amostra</th>
                    <th>x1</th><th>x2</th><th>x3</th>
                    <th>y (T1)</th><th>y (T2)</th><th>y (T3)</th><th>y (T4)</th><th>y (T5)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    test_samples = [
        [-0.3565, 0.0620, 5.9891],
        [-0.7842, 1.1267, 5.5912],
        [0.3012, 0.5611, 5.8234],
        [0.7757, 1.0648, 8.0677],
        [0.1570, 0.8028, 6.3040],
        [-0.7014, 1.0316, 3.6005],
        [0.3748, 0.1536, 6.1537],
        [-0.6920, 0.9404, 4.4058],
        [-1.3970, 0.7141, 4.9263],
        [-1.8842, -0.2805, 1.2548]
    ]
    
    for i, sample in enumerate(test_samples):
        results = test_results[i]
        html_content += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{sample[0]:.4f}</td><td>{sample[1]:.4f}</td><td>{sample[2]:.4f}</td>
                    <td>{results[0]}</td><td>{results[1]}</td><td>{results[2]}</td><td>{results[3]}</td><td>{results[4]}</td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </div>

    <div class="card qa">
        <h2>Análise Teórica</h2>
        <h3>1. Por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron?</h3>
        <p>
            O número de épocas varia porque o vetor de pesos inicial é definido aleatoriamente em cada execução. 
            Como o algoritmo de treinamento (Regra de Hebb Supervisionada) parte de um ponto aleatório no espaço de pesos, 
            a distância e a trajetória necessárias para alcançar uma superfície de decisão que separe perfeitamente as classes 
            mudam a cada vez. Algumas inicializações podem estar "mais próximas" de uma solução válida do que outras, 
            resultando em convergência mais rápida ou mais lenta.
        </p>
        
        <h3>2. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões?</h3>
        <p>
            A principal limitação do Perceptron de camada única é que ele só consegue resolver problemas que são 
            <strong>linearmente separáveis</strong>. Ou seja, ele só pode classificar padrões que podem ser divididos por um hiperplano 
            (uma linha em 2D, um plano em 3D). Se as classes estiverem sobrepostas ou se a fronteira de decisão for não-linear 
            (como no problema do OU-Exclusivo - XOR), o Perceptron nunca convergirá para erro zero.
        </p>
    </div>

    <div class="footer">
        <p>Projeto Perceptron Oil Purity Classifier - 2026</p>
    </div>
</body>
</html>
    """
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✓ report.html gerado com sucesso.")

def main():
    # Load training data
    features, targets = load_data.load_training_data()
    
    test_samples = [
        [-0.3565, 0.0620, 5.9891],
        [-0.7842, 1.1267, 5.5912],
        [0.3012, 0.5611, 5.8234],
        [0.7757, 1.0648, 8.0677],
        [0.1570, 0.8028, 6.3040],
        [-0.7014, 1.0316, 3.6005],
        [0.3748, 0.1536, 6.1537],
        [-0.6920, 0.9404, 4.4058],
        [-1.3970, 0.7141, 4.9263],
        [-1.8842, -0.2805, 1.2548]
    ]
    
    runs_data = []
    test_predictions_all_runs = [[] for _ in range(len(test_samples))]
    
    for run_idx in range(5):
        print(f"\nIniciando Treinamento {run_idx + 1}...")
        
        # Initialize Perceptron with random weights between 0 and 1
        # Re-seeding explicitly if needed, though np.random.rand is usually sufficient
        initial_weights = np.random.rand(4) 
        p = Perceptron(learning_rate=0.01, initial_weights=initial_weights)
        
        errors_per_epoch = []
        max_epochs = 1000
        
        for epoch in range(max_epochs):
            total_errors = 0
            # Shuffle indices to avoid ordering bias (optional but good practice)
            indices = np.arange(len(features))
            np.random.shuffle(indices)
            
            for i in indices:
                x = features[i]
                d = targets[i]
                total_errors += p.train_step(x, d)
            
            errors_per_epoch.append(total_errors)
            
            if total_errors == 0:
                print(f"  Convergência atingida na época {epoch + 1}")
                break
        
        # Save run data
        runs_data.append({
            'initial_weights': initial_weights,
            'final_weights': p.weights.copy(),
            'epochs': len(errors_per_epoch),
            'errors_history': errors_per_epoch
        })
        
        # Test the trained model on test samples
        for s_idx, sample in enumerate(test_samples):
            pred = p.predict(sample)
            test_predictions_all_runs[s_idx].append(pred)
            
        # Plot and save learning curve
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, marker='o', color='red')
        plt.title(f'Treinamento {run_idx + 1}: Erros por Época')
        plt.xlabel('Época')
        plt.ylabel('Número de Erros')
        plt.grid(True)
        plt.savefig(f"error_run_{run_idx + 1}.png")
        plt.close()
        
    # Final Report
    generate_report(runs_data, test_predictions_all_runs)
    print("\nExecução Completa. Abra 'report.html' para ver os resultados.")

if __name__ == "__main__":
    main()
