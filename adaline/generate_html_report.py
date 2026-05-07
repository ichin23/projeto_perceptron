import pandas as pd
import base64
import os

# Função para converter imagem para base64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Caminhos
base_dir = os.path.dirname(__file__)
resultados_treino_path = os.path.join(base_dir, 'resultados.csv')
resultados_teste_path = os.path.join(base_dir, 'resultados_teste.csv')
plot_t1_path = os.path.join(base_dir, 'plots', 'convergencia_T1.png')
plot_t2_path = os.path.join(base_dir, 'plots', 'convergencia_T2.png')
report_path = os.path.join(base_dir, 'report.html')

# Carregar dados
df_treino = pd.read_csv(resultados_treino_path)
df_teste = pd.read_csv(resultados_teste_path)

# Imagens base64
img_t1 = get_base64_image(plot_t1_path)
img_t2 = get_base64_image(plot_t2_path)

html_content = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório ADALINE</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .image-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ccc;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }}
        .answer-box {{
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>

    <h1>Relatório do Projeto ADALINE</h1>
    <p><strong>Aplicação:</strong> Classificação de sinais ruidosos para roteamento de válvulas industriais.</p>

    <h2>1. Resultados dos Treinamentos</h2>
    <p>Cinco treinamentos foram realizados com pesos iniciais aleatórios. Abaixo estão os pesos iniciais, finais e o número de épocas até a convergência.</p>
    
    {df_treino.to_html(index=False, classes='table')}

    <h2>2. Gráficos de Erro Quadrático Médio (T1 e T2)</h2>
    <p>Abaixo estão os gráficos da evolução do Erro Quadrático Médio (MSE) em função do número de épocas para os treinamentos T1 e T2.</p>
    
    <div class="image-container">
        <h3>Treinamento T1</h3>
        <img src="data:image/png;base64,{img_t1}" alt="Gráfico T1">
    </div>

    <div class="image-container">
        <h3>Treinamento T2</h3>
        <img src="data:image/png;base64,{img_t2}" alt="Gráfico T2">
    </div>

    <h2>3. Classificação dos Sinais de Teste</h2>
    <p>A rede ADALINE foi aplicada para classificar os novos sinais. As colunas y (T1) a y (T5) mostram a classificação feita por cada um dos 5 modelos treinados.</p>
    <p><em>-1: Válvula A | 1: Válvula B</em></p>

    {df_teste.to_html(index=False, classes='table')}

    <h2>4. Análise dos Pesos</h2>
    
    <h3>Pergunta:</h3>
    <p><em>Embora o número de épocas de cada treinamento realizado no item 2 seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.</em></p>

    <div class="answer-box">
        <h3>Resposta:</h3>
        <p>A rede ADALINE utiliza a regra de aprendizado de Widrow-Hoff (regra Delta), que busca minimizar a função de custo do Erro Quadrático Médio (MSE). No caso da ADALINE, com uma função de ativação linear durante o treinamento, a superfície de erro é uma <strong>paraboloide multidimensional convexa</strong>.</p>
        
        <p>Isso significa que a função de custo possui um <strong>único mínimo global</strong>. Independentemente de onde os pesos sejam inicializados (como garantido pelas sementes aleatórias nos 5 treinamentos), o algoritmo de gradiente descendente sempre caminhará em direção a este mesmo ponto de mínimo global na superfície de erro.</p>
        
        <p>A diferença no número de épocas ocorre simplesmente porque diferentes pontos de partida iniciais (pesos aleatórios) estão a distâncias diferentes desse mínimo global, exigindo mais ou menos passos para alcançá-lo com a precisão exigida (ε = 10⁻⁶). No entanto, o destino final (os pesos convergidos) será sempre o mesmo, refletindo a solução ótima para o conjunto de dados fornecido.</p>
    </div>

</body>
</html>
"""

with open(report_path, "w", encoding="utf-8") as file:
    file.write(html_content)

print(f"Relatório HTML gerado em: {report_path}")
