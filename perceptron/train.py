import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from perceptron import Perceptron
import load_data
import json
import os

def generate_premium_report(runs_data, test_results):
    """
    Generates a stunning, modern report.html using glassmorphism and interactive Plotly charts.
    """
    
    # Generate Plotly charts as HTML snippets
    charts_html = []
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
    
    for i, run in enumerate(runs_data):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(run['errors_history']) + 1)),
            y=run['errors_history'],
            mode='lines+markers',
            name=f'Run {i+1}',
            line=dict(color=colors[i], width=3),
            marker=dict(size=8, symbol='circle')
        ))
        
        fig.update_layout(
            title=f'Curva de Aprendizado - Treinamento {i+1}',
            xaxis_title='Época',
            yaxis_title='Número de Erros',
            template='plotly_white',
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", size=12)
        )
        
        # Convert to HTML snippet
        chart_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        charts_html.append(chart_div)

    # Prepare tables
    runs_table_rows = ""
    for i, run in enumerate(runs_data):
        wi = run['initial_weights']
        wf = run['final_weights']
        runs_table_rows += f"""
            <tr>
                <td class="run-id">T{i+1}</td>
                <td><span class="weight-tag">w0: {wi[0]:.4f}</span><br>w1: {wi[1]:.4f}<br>w2: {wi[2]:.4f}<br>w3: {wi[3]:.4f}</td>
                <td><span class="weight-tag final">w0: {wf[0]:.4f}</span><br>w1: {wf[1]:.4f}<br>w2: {wf[2]:.4f}<br>w3: {wf[3]:.4f}</td>
                <td class="epoch-count">{run['epochs']}</td>
            </tr>
        """

    test_samples = [
        [-0.3565, 0.0620, 5.9891], [-0.7842, 1.1267, 5.5912], [0.3012, 0.5611, 5.8234],
        [0.7757, 1.0648, 8.0677], [0.1570, 0.8028, 6.3040], [-0.7014, 1.0316, 3.6005],
        [0.3748, 0.1536, 6.1537], [-0.6920, 0.9404, 4.4058], [-1.3970, 0.7141, 4.9263],
        [-1.8842, -0.2805, 1.2548]
    ]
    
    test_table_rows = ""
    for i, sample in enumerate(test_samples):
        results = test_results[i]
        res_cells = "".join([f'<td class="class-res c{"pos" if r > 0 else "neg"}">{r}</td>' for r in results])
        test_table_rows += f"""
            <tr>
                <td class="sample-id">{i+1}</td>
                <td class="coords">{sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}</td>
                {res_cells}
            </tr>
        """

    html_template = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Insights | Perceptron Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #636EFA;
            --secondary: #EF553B;
            --accent: #00CC96;
            --bg: #0f172a;
            --glass: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.1);
            --text: #f8fafc;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Inter', sans-serif; 
            background: radial-gradient(circle at top right, #1e293b, #0f172a); 
            color: var(--text);
            padding: 40px 20px;
            line-height: 1.6;
        }}

        .container {{ max-width: 1200px; margin: 0 auto; }}

        header {{ 
            text-align: center; 
            margin-bottom: 60px; 
            animation: fadeInDown 1s ease-out;
        }}

        h1 {{ 
            font-family: 'Outfit', sans-serif;
            font-size: 3.5rem; 
            font-weight: 800;
            background: linear-gradient(to right, #818CF8, #34D399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
            filter: drop-shadow(0 2px 10px rgba(99, 110, 250, 0.3));
        }}

        .subtitle {{ font-size: 1.2rem; color: #cbd5e1; letter-spacing: 2px; text-transform: uppercase; font-weight: 600; }}

        .grid {{ 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); 
            gap: 30px; 
            margin-bottom: 50px;
        }}

        .card {{
            background: var(--glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            padding: 30px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }}

        h2 {{ 
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem; 
            margin-bottom: 25px; 
            display: flex; 
            align-items: center;
            gap: 12px;
            color: #ffffff;
            letter-spacing: -0.5px;
        }}

        h2::before {{
            content: '';
            width: 8px;
            height: 24px;
            background: var(--primary);
            border-radius: 4px;
        }}

        table {{ width: 100%; border-collapse: separate; border-spacing: 0 8px; }}
        th {{ text-align: left; padding: 15px; color: #cbd5e1; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; }}
        td {{ padding: 15px; background: rgba(255,255,255,0.03); }}
        tr td:first-child {{ border-radius: 12px 0 0 12px; }}
        tr td:last-child {{ border-radius: 0 12px 12px 0; }}

        .run-id {{ font-weight: 700; color: var(--primary); }}
        .weight-tag {{ font-size: 0.85rem; color: #94a3b8; }}
        .weight-tag.final {{ color: var(--accent); }}
        .epoch-count {{ font-size: 1.2rem; font-weight: 700; color: #fff; }}

        .charts-section {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 50px; }}
        .chart-card {{ padding: 15px; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); }}

        .class-res {{ font-weight: 800; border-radius: 8px !important; }}
        .cpos {{ color: var(--accent); background: rgba(0, 204, 150, 0.1) !important; }}
        .cneg {{ color: var(--secondary); background: rgba(239, 85, 59, 0.1) !important; }}

        .qa-section {{ background: linear-gradient(135deg, rgba(99, 110, 250, 0.1), rgba(0, 204, 150, 0.1)); }}
        .question {{ margin-bottom: 30px; }}
        .question h3 {{ font-size: 1.4rem; margin-bottom: 12px; color: #fff; }}
        .question p {{ color: #cbd5e1; font-size: 1.1rem; line-height: 1.8; }}

        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .footer {{ text-align: center; color: #64748b; padding-top: 50px; border-top: 1px solid var(--glass-border); margin-top: 100px; }}

        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: var(--bg); }}
        ::-webkit-scrollbar-thumb {{ background: var(--glass-border); border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="subtitle">Neural Network Dashboard</div>
            <h1>Perceptron Oil Insights</h1>
        </header>

        <div class="grid">
            <div class="card">
                <h2>Performance dos Treinamentos</h2>
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Pesos Iniciais</th>
                            <th>Pesos Finais</th>
                            <th>Épocas</th>
                        </tr>
                    </thead>
                    <tbody>
                        {runs_table_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Resultados de Classificação</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Amostra</th>
                            <th>Coordenadas</th>
                            <th>T1</th><th>T2</th><th>T3</th><th>T4</th><th>T5</th>
                        </tr>
                    </thead>
                    <tbody>
                        {test_table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="card charts-section">
            {charts_html[0]}
            {charts_html[1]}
            {charts_html[2]}
            {charts_html[3]}
            {charts_html[4]}
        </div>

        <div class="card qa-section">
            <h2>Análise de Deep Learning</h2>
            <div class="question">
                <h3>Variação do Número de Épocas</h3>
                <p>
                    A variabilidade observada no número de épocas (de {min([r['epochs'] for r in runs_data])} a {max([r['epochs'] for r in runs_data])}) é um reflexo direto da estocasticidade na inicialização. No espaço de alta dimensionalidade dos pesos, a "paisagem" de erro contém múltiplas trajetórias para a convergência. Ao sortear pesos entre 0 e 1, posicionamos o modelo em diferentes "regiões de largada". Algumas dessas regiões possuem um gradiente de erro mais favorável ou estão geometricamente mais próximas de uma fronteira de decisão válida, permitindo que a Regra de Hebb Supervisionada ajuste o hiperplano em menos iterações.
                </p>
            </div>
            <div class="question">
                <h3>Limitações Fundamentais</h3>
                <p>
                    O Perceptron é o bloco fundamental das redes neurais, mas sua arquitetura de camada única impõe uma restrição geométrica severa: a <strong>Linear Separability</strong>. Ele assume que o mundo pode ser dividido por uma régua reta. Em cenários reais de engenharia química ou geofísica, onde as propriedades físico-químicas do petróleo podem interagir de forma não-linear, o Perceptron pode falhar. Para superar isso, seriam necessárias camadas ocultas (MLP) com funções de ativação não-lineares, permitindo a criação de superfícies de decisão complexas e curvas.
                </p>
            </div>
        </div>

        <div class="footer">
            <p>Generated by Antigravity AI Core &bull; CEFET Project &bull; 2026</p>
        </div>
    </div>
</body>
</html>
    """
    
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_template)
    print("✓ report_premium.html gerado com sucesso.")

def main():
    print("🚀 Iniciando Motor de Treinamento Premium...")
    
    # Load training data
    features, targets = load_data.load_training_data()
    
    test_samples = [
        [-0.3565, 0.0620, 5.9891], [-0.7842, 1.1267, 5.5912], [0.3012, 0.5611, 5.8234],
        [0.7757, 1.0648, 8.0677], [0.1570, 0.8028, 6.3040], [-0.7014, 1.0316, 3.6005],
        [0.3748, 0.1536, 6.1537], [-0.6920, 0.9404, 4.4058], [-1.3970, 0.7141, 4.9263],
        [-1.8842, -0.2805, 1.2548]
    ]
    
    runs_data = []
    test_predictions_all_runs = [[] for _ in range(len(test_samples))]
    
    for run_idx in range(5):
        # Initialize Perceptron with random weights between 0 and 1
        initial_weights = np.random.rand(4) 
        p = Perceptron(learning_rate=0.01, initial_weights=initial_weights)
        
        errors_history = []
        max_epochs = 1000
        
        for epoch in range(max_epochs):
            total_errors = 0
            indices = np.arange(len(features))
            np.random.shuffle(indices)
            
            for i in indices:
                total_errors += p.train_step(features[i], targets[i])
            
            errors_history.append(total_errors)
            if total_errors == 0: break
        
        runs_data.append({
            'initial_weights': initial_weights,
            'final_weights': p.weights.copy(),
            'epochs': len(errors_history),
            'errors_history': errors_history
        })
        
        for s_idx, sample in enumerate(test_samples):
            test_predictions_all_runs[s_idx].append(p.predict(sample))
            
    # Final Report
    generate_premium_report(runs_data, test_predictions_all_runs)
    print("\n✨ DASHBOARD CONSTRUÍDO COM SUCESSO. Visualize report.html.")

if __name__ == "__main__":
    main()
