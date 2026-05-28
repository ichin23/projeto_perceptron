import os
import json
import pandas as pd

# ─── Configurações ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, 'report.html')
INDEX_PATH = os.path.join(BASE_DIR, 'index.html')

def generate_grid_html(original, noisy, recovered, flips, success):
    """
    Generates HTML representation of 9x5 grids for original, noisy, and recovered states.
    """
    def get_cells_html(vals, is_noisy_grid=False):
        html = ""
        for idx, val in enumerate(vals):
            cell_class = "grid-cell"
            title_text = f"Pixel {idx}"
            
            if val == 1:
                cell_class += " dark-pixel"
            else:
                cell_class += " white-pixel"
                
            if is_noisy_grid and idx in flips:
                cell_class += " corrupted-pixel"
                title_text += " (CORROMPIDO)"
                
            html += f'<div class="{cell_class}" title="{title_text}"></div>'
        return html

    status_tag = '<span class="status-badge success">✓ SUCESSO</span>' if success else '<span class="status-badge failure">✗ FALHA</span>'

    html = f"""
    <div class="digit-comparison-card">
        <div class="grids-wrapper">
            <div class="grid-column">
                <h4>Original</h4>
                <div class="grid-container">
                    {get_cells_html(original)}
                </div>
            </div>
            <div class="grid-arrow">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                <span class="arrow-label">20% ruído</span>
            </div>
            <div class="grid-column">
                <h4>Decodificado (Link)</h4>
                <div class="grid-container">
                    {get_cells_html(noisy, is_noisy_grid=True)}
                </div>
            </div>
            <div class="grid-arrow">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                <span class="arrow-label">Rede Hopfield</span>
            </div>
            <div class="grid-column">
                <h4>Recuperado</h4>
                <div class="grid-container">
                    {get_cells_html(recovered)}
                </div>
            </div>
        </div>
        <div class="digit-metadata">
            {status_tag}
        </div>
    </div>
    """
    return html

def build_report():
    print("Gerando relatório HTML...")
    
    # Load data
    df_sens = pd.read_csv(os.path.join(BASE_DIR, 'resultados_sensibilidade.csv'))
    df_20 = pd.read_csv(os.path.join(BASE_DIR, 'resultados_20_ruido.csv'))
    
    with open(os.path.join(BASE_DIR, 'sample_runs.json'), 'r', encoding='utf-8') as f:
        sample_runs = json.load(f)
        
    # Generate HTML grids for each digit, displaying all 3 situations
    grids_html_list = []
    for name in ["1", "2", "3", "4"]:
        runs_list = sample_runs[name]
        
        runs_html = ""
        for i, run in enumerate(runs_list):
            runs_html += f"""
            <div class="situation-container">
                <h4 class="situation-title">Situação de Transmissão {name}.{i+1}</h4>
                {generate_grid_html(run['original'], run['noisy'], run['recovered_async'], run['flips'], run['success_async'])}
                <div class="sample-stats">
                    <div class="stat-box">
                        <span class="stat-label">Convergência Asíncrona</span>
                        <span class="stat-value">{run['history_async_sweeps']} sweeps</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Energia Inicial</span>
                        <span class="stat-value">{run['history_async_energies'][0]:.3f}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Energia Final</span>
                        <span class="stat-value">{run['history_async_energies'][-1]:.3f}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Dinâmica Síncrona</span>
                        <span class="stat-value">{"Sim" if run['converged_sync'] else "Não"} ({run['history_sync_iterations']} iters)</span>
                    </div>
                </div>
            </div>
            """
            
        digit_section_html = f"""
        <div class="digit-section">
            <h3>Dígito Padrão {name}</h3>
            <div class="situations-wrapper">
                {runs_html}
            </div>
        </div>
        """
        grids_html_list.append(digit_section_html)
        
    grids_combined_html = "\n".join(grids_html_list)
    
    # Prepare tables for display
    results_20_table = ""
    for _, r in df_20.iterrows():
        results_20_table += f"""
        <tr>
            <td><strong>Dígito {r['Digito']}</strong></td>
            <td><span class="rate-badge async">{r['Taxa_Async']:.1f}%</span></td>
            <td>{r['Média_Sweeps_Async']:.2f}</td>
            <td><span class="rate-badge sync">{r['Taxa_Sync']:.1f}%</span></td>
            <td>{r['Média_Iters_Sync']:.2f}</td>
        </tr>
        """
        
    # JSON data for charts injection
    sensitivity_json = df_sens.to_json(orient='records')
    sample_runs_json = json.dumps(sample_runs)
    
    html_content = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório - Memória Associativa de Hopfield</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {{
            --bg-color: #0f172a;
            --text-color: #f8fafc;
            --text-secondary: #94a3b8;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Outfit', sans-serif;
        }}

        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
            background-attachment: fixed;
            line-height: 1.6;
            padding-bottom: 5rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            margin-bottom: 4rem;
            animation: fadeInDown 1s ease-out;
            position: relative;
        }}

        .back-btn {{
            position: absolute;
            left: 0;
            top: 50%;
            transform: translateY(-50%);
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-color);
            text-decoration: none;
            background: rgba(255, 255, 255, 0.05);
            padding: 0.75rem 1.25rem;
            border-radius: 999px;
            border: 1px solid var(--card-border);
            font-weight: 500;
            transition: all 0.3s ease;
            backdrop-filter: blur(8px);
        }}

        .back-btn:hover {{
            background: rgba(255, 255, 255, 0.15);
            transform: translateY(-50%) translateX(-4px);
            border-color: var(--accent-blue);
        }}

        .back-btn svg {{
            width: 20px;
            height: 20px;
        }}

        h1 {{
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(to right, #60a5fa, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            font-size: 1.2rem;
            color: var(--text-secondary);
            font-weight: 300;
        }}

        .grid-layout {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 2rem;
            margin-bottom: 3rem;
        }}

        @media (max-width: 968px) {{
            .grid-layout {{
                grid-template-columns: 1fr;
            }}
            header {{
                padding-top: 4rem;
            }}
            .back-btn {{
                top: 0;
                transform: none;
            }}
            .back-btn:hover {{
                transform: translateX(-4px);
            }}
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 2.5rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            margin-bottom: 2rem;
        }}

        .card.full-width {{
            grid-column: 1 / -1;
        }}

        h2 {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            border-left: 4px solid var(--accent-blue);
            padding-left: 0.75rem;
        }}

        h3 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            color: #93c5fd;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.5rem;
        }}

        p {{
            color: #cbd5e1;
            margin-bottom: 1.25rem;
        }}

        ul, ol {{
            margin-left: 2rem;
            margin-bottom: 1.5rem;
            color: #cbd5e1;
        }}

        li {{
            margin-bottom: 0.5rem;
        }}

        /* Grids Styling */
        .digit-section {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .situation-container {{
            background: rgba(255, 255, 255, 0.015);
            border: 1px solid rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .situation-title {{
            color: #93c5fd !important;
            font-size: 1.05rem !important;
            margin-bottom: 0.75rem !important;
            font-weight: 600 !important;
            border-bottom: none !important;
            padding-bottom: 0 !important;
        }}

        .digit-comparison-card {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1.5rem;
            margin: 1rem 0;
        }}

        .grids-wrapper {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 2rem;
            flex-wrap: wrap;
            width: 100%;
        }}

        .grid-column {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
        }}

        .grid-column h4 {{
            font-size: 0.9rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .grid-container {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 4px;
            width: 120px;
            background: #1e293b;
            padding: 8px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .grid-cell {{
            width: 18px;
            height: 18px;
            border-radius: 3px;
            transition: all 0.3s ease;
        }}

        .grid-cell.white-pixel {{
            background: #e2e8f0;
        }}

        .grid-cell.dark-pixel {{
            background: #0f172a;
            border: 1px solid rgba(255,255,255,0.05);
        }}

        .grid-cell.corrupted-pixel {{
            box-shadow: 0 0 8px #ef4444;
            border: 1px solid #ef4444;
        }}

        .grid-arrow {{
            display: flex;
            flex-direction: column;
            align-items: center;
            color: var(--accent-blue);
        }}

        .grid-arrow svg {{
            width: 32px;
            height: 32px;
            animation: pulseHorizontal 1.5s infinite ease-in-out;
        }}

        .arrow-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }}

        .digit-metadata {{
            text-align: center;
        }}

        .status-badge {{
            padding: 0.4rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }}

        .status-badge.success {{
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }}

        .status-badge.failure {{
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }}

        .sample-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1.5rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            padding: 1rem;
        }}

        .stat-box {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}

        .stat-label {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }}

        .stat-value {{
            font-size: 1.1rem;
            font-weight: 600;
            color: #93c5fd;
        }}

        /* Table Styling */
        .table-container {{
            overflow-x: auto;
            margin-top: 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--card-border);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            background: rgba(255, 255, 255, 0.01);
        }}

        th, td {{
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--card-border);
        }}

        th {{
            background: rgba(255, 255, 255, 0.05);
            font-weight: 600;
            color: #93c5fd;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        .rate-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-weight: 600;
            font-size: 0.85rem;
        }}

        .rate-badge.async {{
            background: rgba(59, 130, 246, 0.15);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}

        .rate-badge.sync {{
            background: rgba(139, 92, 246, 0.15);
            color: #c084fc;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }}

        /* Plotly Containers */
        .chart-container {{
            width: 100%;
            height: 400px;
            margin-top: 1.5rem;
            background: rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            border: 1px solid var(--card-border);
            padding: 1rem;
        }}

        .math-block {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 1rem;
            font-family: monospace;
            overflow-x: auto;
            margin: 1rem 0;
            color: #a7f3d0;
        }}

        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translateY(-30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes pulseHorizontal {{
            0%, 100% {{
                transform: translateX(0);
            }}
            50% {{
                transform: translateX(4px);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <a href="../index.html" class="back-btn">
                <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                Voltar ao Portfólio
            </a>
            <h1>Projeto Hopfield</h1>
            <p class="subtitle">Memória Associativa de 45 Neurônios para Recuperação de Imagens Transmitidas sob Ruído</p>
        </header>

        <div class="grid-layout">
            <!-- Coluna da Esquerda: Teoria e Conceitos -->
            <div class="card">
                <h2>Fundamentação Teórica</h2>
                <p>
                    A rede de <strong>Hopfield</strong> é uma rede neural recorrente de camada única que atua como uma <strong>memória associativa auto-associativa</strong>. Diferente das redes tradicionais feed-forward (como Perceptron e ADALINE), ela armazena informações como pontos de equilíbrio estáveis (atratores) de um sistema dinâmico.
                </p>
                <p>
                    O modelo consiste em 45 neurônios dispostos em uma grade de 9x5 pixels. A representação dos estados é bipolar: <strong>Branco (-1)</strong> e <strong>Escuro (+1)</strong>.
                </p>

                <h3>1. Regra de Aprendizado Hebbiano</h3>
                <p>
                    Os pesos sinápticos são determinados de forma direta (sem iterações de treino baseadas em gradiente) a partir da média dos produtos externos dos padrões que desejamos armazenar:
                </p>
                <div class="math-block">
                    W = (1 / N) * sum_{{&mu;}} (x^{{&mu;}} * (x^{{&mu;}})^T)<br>
                    W_ii = 0 (sem auto-conexão)
                </div>
                <p>
                    Onde <em>N = 45</em> é o número de neurônios, e <em>x^{{&mu;}}</em> representa o vetor do padrão &mu;. Zerar a diagonal impede que o neurônio reforce indefinidamente o seu próprio estado anterior, o que favorece a convergência para padrões gerais.
                </p>

                <h3>2. Dinâmica de Atualização</h3>
                <p>
                    Implementamos duas abordagens de atualização:
                </p>
                <ul>
                    <li><strong>Asíncrona (Padrão)</strong>: Um único neurônio é selecionado aleatoriamente e atualizado a cada instante. Isso garante que a <strong>função de energia Lyapunov</strong> decresça de forma estritamente monótona, atingindo um mínimo local estável.</li>
                    <li><strong>Síncrona</strong>: Todos os neurônios são atualizados ao mesmo tempo. Embora mais rápida, essa dinâmica pode levar a oscilações cíclicas (período 2) ao invés de atratores pontuais fixos.</li>
                </ul>
            </div>

            <!-- Coluna da Direita: Parâmetros e Resultados Globais -->
            <div class="card">
                <h2>Parâmetros e Configuração</h2>
                <ul>
                    <li><strong>Grade de Neurônios:</strong> 45 (9 linhas x 5 colunas)</li>
                    <li><strong>Padrões Armazenados:</strong> 4 dígitos ("1", "2", "3", "4")</li>
                    <li><strong>Link de Comunicação:</strong> Injeção de 20% de ruído pseudo-aleatório (9 pixels invertidos por teste)</li>
                    <li><strong>Simulações por Dígito:</strong> 1000 repetições estocásticas</li>
                    <li><strong>Análise de Sensibilidade:</strong> Ruído variável de 0% a 50% em passos de 5%</li>
                </ul>

                <h2 style="margin-top: 2rem;">Desempenho Geral (20% de Ruído)</h2>
                <p>Resultados compilados após 1000 transmissões estocásticas por padrão:</p>
                
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Padrão</th>
                                <th>Sucesso Async</th>
                                <th>Sweeps Médios</th>
                                <th>Sucesso Sync</th>
                                <th>Iters Média</th>
                            </tr>
                        </thead>
                        <tbody>
                            {results_20_table}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Linha Completa: Efeito do Ruído Excessivo -->
            <div class="card full-width">
                <h2>Efeito do Ruído Excessivo na Rede de Hopfield</h2>
                <p>
                    A estabilidade e robustez de uma rede de Hopfield dependem da distância (em termos de distância de Hamming) entre o estado de entrada e o atrator correspondente. Quando aumentamos excessivamente o nível de ruído (geralmente acima de 30% a 40% de pixels corrompidos), ocorrem três fenômenos principais:
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.5rem; margin-top: 1.5rem;">
                    <div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 12px; padding: 1.5rem;">
                        <h3 style="color: #f87171; margin-bottom: 0.5rem; border-bottom: none; padding-bottom: 0;">1. Convergência para Estados Espúrios</h3>
                        <p style="font-size: 0.9rem; margin-bottom: 0;">
                            A regra Hebbiana de treinamento gera atratores indesejados chamados de <strong>estados espúrios</strong>, que são mínimos locais da função de energia que não fazem parte do conjunto de treinamento. Com ruído extremo, a rede perde a referência do padrão original e converge para essas combinações lineares ou misturas dos padrões originais.
                        </p>
                    </div>
                    <div style="background: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.2); border-radius: 12px; padding: 1.5rem;">
                        <h3 style="color: #fbbf24; margin-bottom: 0.5rem; border-bottom: none; padding-bottom: 0;">2. Atração por Outros Padrões</h3>
                        <p style="font-size: 0.9rem; margin-bottom: 0;">
                            Se o ruído alterar pixels chaves que tornam a imagem mais similar a outro dígito do que ao original (ex: corromper o dígito 3 de modo que ele se pareça com o dígito 2), a dinâmica da rede empurrará o estado para a bacia de atração do padrão concorrente, resultando na recuperação de uma imagem limpa, porém incorreta.
                        </p>
                    </div>
                    <div style="background: rgba(59, 130, 246, 0.05); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 1.5rem;">
                        <h3 style="color: #60a5fa; margin-bottom: 0.5rem; border-bottom: none; padding-bottom: 0;">3. Estados Complementares (Inversos)</h3>
                        <p style="font-size: 0.9rem; margin-bottom: 0;">
                            Devido à simetria dos pesos (W_ij = W_ji), se um padrão <em>x</em> é estável, o seu inverso <em>-x</em> também o é (uma imagem em negativo fotográfico). Se o ruído ultrapassar 50%, a entrada estará mais próxima do inverso do padrão original, fazendo com que a rede recupere a imagem com cores completamente invertidas.
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Linha Completa: Demonstração Visual das Grades (12 situações) -->
            <div class="card full-width">
                <h2>Simulação de 12 Situações de Transmissão (3 para cada padrão)</h2>
                <p>
                    Abaixo apresentamos as saídas reais de amostras de transmissão no link de comunicação. Os pixels em <strong>vermelho destacado</strong> foram corrompidos no link de comunicação e recuperados pela rede de Hopfield assíncrona.
                </p>
                {grids_combined_html}
            </div>

            <!-- Linha Completa: Gráficos de Sensibilidade e Energia -->
            <div class="card full-width">
                <h2>Análise de Robustez e Conectividade Sináptica</h2>
                <p>
                    Explore abaixo os gráficos interativos gerados para validar a rede. A curva de sensibilidade demonstra o impacto da densidade de ruído na taxa de acerto. A trajetória de energia prova o decaimento estável do estado energético durante o processo assíncrono.
                </p>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                    <div>
                        <h3>Taxa de Recuperação vs. Nível de Ruído</h3>
                        <div id="sensitivity-chart" class="chart-container"></div>
                    </div>
                    <div>
                        <h3>Dinâmica de Energia (Estabilidade Lyapunov)</h3>
                        <div id="energy-chart" class="chart-container"></div>
                        <div style="margin-top: 1rem; text-align: center;">
                            <label for="digit-selector" style="font-size: 0.9rem; color: var(--text-secondary); margin-right: 0.5rem;">Selecione o Dígito:</label>
                            <select id="digit-selector" style="background: #1e293b; color: white; border: 1px solid var(--card-border); padding: 0.4rem 1rem; border-radius: 8px;">
                                <option value="1">Dígito 1</option>
                                <option value="2">Dígito 2</option>
                                <option value="3">Dígito 3</option>
                                <option value="4">Dígito 4</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Linha Completa: Discussão e Conclusões -->
            <div class="card full-width">
                <h2>Discussão e Conclusão dos Resultados</h2>
                <p>
                    A análise das simulações revela características essenciais da rede de Hopfield como memória associativa:
                </p>
                <ol>
                    <li>
                        <strong>Capacidade de Armazenamento e Ortogonalidade</strong>: 
                        O dígito "1" obteve a melhor taxa de recuperação (~91.5% sob 20% de ruído) devido à sua baixa correlação com os demais padrões (ex: produto interno normalizado de apenas -0.20 com o dígito "4").
                        Por outro lado, o dígito "3" apresentou a menor taxa de recuperação (~45.8%). Isso se deve à sua grande similaridade estrutural com o dígito "2" (correlação de +0.47, ou 21 pixels correspondentes de mesma cor). A alta correlação cria bacias de atração sobrepostas e estados espúrios indesejados, confundindo a rede em níveis maiores de ruído.
                    </li>
                    <li>
                        <strong>Estabilidade Lyapunov</strong>:
                        O gráfico da dinâmica de energia demonstra de forma explícita que a energia da rede decresce de maneira monótona a cada atualização de neurônio (asíncrona), alcançando um patamar estável mínimo (atrator local).
                    </li>
                    <li>
                        <strong>Comparações de Atualização</strong>:
                        A dinâmica assíncrona mostrou-se ligeiramente mais resiliente e livre de oscilações na maioria dos casos em comparação com a dinâmica síncrona. Esta última ocasionalmente oscila entre dois estados complementares, não convergindo a um padrão fixo estável.
                    </li>
                </ol>
            </div>
        </div>
    </div>

    <script>
        // Injected data from Python simulation runs
        const sensData = {sensitivity_json};
        const sampleRuns = {sample_runs_json};

        // --- Render Sensitivity Chart ---
        const noiseLevels = sensData.map(d => parseInt(d['Ruído']));
        const sensitivityTraces = [];
        const digits = ["1", "2", "3", "4"];
        const colorsAsync = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'];
        const colorsSync = ['#60a5fa', '#34d399', '#fbbf24', '#a78bfa'];

        digits.forEach((name, i) => {{
            // Async trace
            sensitivityTraces.push({{
                x: noiseLevels,
                y: sensData.map(d => d[`Async_D${{name}}`] * 100),
                name: `Dígito ${{name}} (Async)`,
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: colorsAsync[i], width: 2.5 }},
                marker: {{ size: 6 }}
            }});
            // Sync trace
            sensitivityTraces.push({{
                x: noiseLevels,
                y: sensData.map(d => d[`Sync_D${{name}}`] * 100),
                name: `Dígito ${{name}} (Sync)`,
                type: 'scatter',
                mode: 'lines',
                line: {{ color: colorsSync[i], width: 1.5, dash: 'dash' }},
                opacity: 0.6
            }});
        }});

        const sensLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#f8fafc', family: 'Outfit, sans-serif' }},
            margin: {{ t: 20, r: 20, l: 50, b: 50 }},
            xaxis: {{ title: 'Porcentagem de Ruído (%)', gridcolor: 'rgba(255,255,255,0.05)', zeroline: false }},
            yaxis: {{ title: 'Taxa de Recuperação (%)', gridcolor: 'rgba(255,255,255,0.05)', range: [-5, 105] }},
            legend: {{ orientation: 'h', y: -0.2, x: 0 }},
            hovermode: 'x unified'
        }};

        Plotly.newPlot('sensitivity-chart', sensitivityTraces, sensLayout, {{ responsive: true, displayModeBar: false }});

        // --- Render Energy Chart ---
        function updateEnergyChart(digitName) {{
            const runs = sampleRuns[digitName];
            // Render for the first scenario of the selected digit
            const firstRun = runs[0];
            const asyncEnergies = firstRun.history_async_energies;
            const syncEnergies = firstRun.history_sync_energies;

            const energyTraces = [
                {{
                    x: Array.from({{ length: asyncEnergies.length }}, (_, i) => i),
                    y: asyncEnergies,
                    name: 'Atualização Assíncrona (Cenário 1)',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {{ color: '#3b82f6', width: 2 }},
                    marker: {{ size: 4 }}
                }},
                {{
                    x: Array.from({{ length: syncEnergies.length }}, (_, i) => i * 45),
                    y: syncEnergies,
                    name: 'Atualização Síncrona (Cenário 1, Escalada)',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#8b5cf6', width: 1.5, dash: 'dash' }}
                }}
            ];

            const energyLayout = {{
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#f8fafc', family: 'Outfit, sans-serif' }},
                margin: {{ t: 20, r: 20, l: 50, b: 50 }},
                xaxis: {{ title: 'Micro-passos (Atualizações de Neurônio)', gridcolor: 'rgba(255,255,255,0.05)', zeroline: false }},
                yaxis: {{ title: 'Energia E(s)', gridcolor: 'rgba(255,255,255,0.05)' }},
                legend: {{ orientation: 'h', y: -0.2, x: 0 }},
                hovermode: 'x unified'
            }};

            Plotly.newPlot('energy-chart', energyTraces, energyLayout, {{ responsive: true, displayModeBar: false }});
        }}

        // Initialize energy chart for Digit 1
        updateEnergyChart("1");

        // Event listener for digit selection change
        document.getElementById('digit-selector').addEventListener('change', (e) => {{
            updateEnergyChart(e.target.value);
        }});
    </script>
</body>
</html>
"""
    
    # Write to files
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Relatório tempo real salvo em: {REPORT_PATH}")
    
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Relatório duplicado em index.html (para roteamento Vercel): {INDEX_PATH}")
    
if __name__ == '__main__':
    build_report()
