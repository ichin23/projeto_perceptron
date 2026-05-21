# Projeto RBF — Classificador de Radiação (rdl1)

Este diretório contém a implementação de uma Rede Neural de Funções de Base Radial (RBF) de 2 entradas e 1 saída para a classificação da presença ($1.0$) ou ausência ($-1.0$) de radiação em compostos nucleares com base em duas variáveis de concentração ($x_1$ e $x_2$).

A base de dados é composta por 50 amostras (40 para treinamento fornecidas pelo usuário e 10 para teste/avaliação retiradas da literatura padrão do projeto 6.5 do livro *"Redes Neurais Artificiais para Engenharia e Ciências Aplicadas"* de Ivan Nunes da Silva).

---

## Resultados das Questões

### 1. Treinamento da Camada Escondida (K-Means com Classe $d = 1.0$)
Para a camada oculta, computou-se os centros de **dois** clusters ($K = 2$) levando em consideração apenas os 19 padrões de treinamento correspondentes à **presença de radiação** ($d = 1.0$). Os centroides (coordenadas) e variâncias resultantes são:

*   **Cluster 1 (6 padrões):**
    *   **Centro $(x_1, x_2)$:** $(0.164833, 0.612117)$
    *   **Variância Populacional (Biased):** $Var(x_1) = 0.007635$, $Var(x_2) = 0.022170$ | **Total:** $0.029806$
    *   **Variância Amostral (Unbiased):** $Var(x_1) = 0.009162$, $Var(x_2) = 0.026605$ | **Total:** $0.035767$
*   **Cluster 2 (13 padrões):**
    *   **Centro $(x_1, x_2)$:** $(0.398969, 0.157131)$
    *   **Variância Populacional (Biased):** $Var(x_1) = 0.027646$, $Var(x_2) = 0.010814$ | **Total:** $0.038460$
    *   **Variância Amostral (Unbiased):** $Var(x_1) = 0.029949$, $Var(x_2) = 0.011715$ | **Total:** $0.041665$

---

### 2. Treinamento da Camada de Saída (Regra Delta)
Após fixar os centros, treinou-se o neurônio linear da camada de saída com a **Regra Delta Generalizada (LMS)** sob taxa de aprendizado $\eta = 0.01$ e precisão $\epsilon = 10^{-7}$. A tabela abaixo apresenta os pesos convergidos para cada estratégia de espalhamento ($\sigma_j^2$):

| Peso | Variância Populacional (Biased) | Variância Amostral (Unbiased) | Espalhamento Heurístico (Distância) |
| :--- | :---: | :---: | :---: |
| **$W_{21,0}$** (Bias) | **$-0.989750$** | $-1.037252$ | $-1.226912$ |
| **$W_{21,1}$** (Hidden 1) | **$2.337761$** | $2.145476$ | $1.549965$ |
| **$W_{21,2}$** (Hidden 2) | **$2.685491$** | $2.665531$ | $2.485053$ |
| **Épocas** | $9.316$ | $8.647$ | $6.986$ |
| **MSE Final** | $0.234333$ | $0.236506$ | $0.255678$ |

*Nota: A solução analítica ótima de Mínimos Quadrados fornece $W_{21,0} = -1.001231$, $W_{21,1} = 2.384409$ e $W_{21,2} = 2.705125$.*

---

### 3. Rotina de Pós-Processamento (Função Sinal)
Para a conversão das saídas reais lineares da rede ($y$) para classes binárias discretas ($y_{\text{pós}} \in \{-1, 1\}$) no conjunto de teste, aplica-se a função sinal:
$$y_{\text{pós}} = \begin{cases} 1, & \text{se } y \geq 0 \\ -1, & \text{se } y < 0 \end{cases}$$

Implementada em Python como:
```python
y_pos = np.where(y_raw >= 0, 1.0, -1.0)
```

---

### 4. Validação no Conjunto de Teste
A validação da rede sobre os 10 padrões de teste produziu os seguintes resultados de saída para as diferentes parametrizações:

#### Caso 1: Espalhamento via Variância Populacional (Acurácia: 80.0%)
*   **Amostra 1** (0.8705, 0.9329): $y = -0.989592 \implies y_{\text{pós}} = -1.0$ (Desejado: $-1.0$) — **ACERTO**
*   **Amostra 2** (0.0388, 0.2703): $y = -0.316465 \implies y_{\text{pós}} = -1.0$ (Desejado: $1.0$) — **ERRO**
*   **Amostra 3** (0.8236, 0.4458): $y = -0.901542 \implies y_{\text{pós}} = -1.0$ (Desejado: $-1.0$) — **ACERTO**
*   **Amostra 4** (0.7075, 0.1502): $y = -0.210715 \implies y_{\text{pós}} = -1.0$ (Desejado: $1.0$) — **ERRO**
*   **Amostra 5** (0.9587, 0.8663): $y = -0.989664 \implies y_{\text{pós}} = -1.0$ (Desejado: $-1.0$) — **ACERTO**
*   **Amostra 6** (0.6115, 0.9365): $y = -0.975112 \implies y_{\text{pós}} = -1.0$ (Desejado: $-1.0$) — **ACERTO**
*   **Amostra 7** (0.3534, 0.3646): $y = 0.964685 \implies y_{\text{pós}} = 1.0$ (Desejado: $1.0$) — **ACERTO**
*   **Amostra 8** (0.3268, 0.2766): $y = 1.322695 \implies y_{\text{pós}} = 1.0$ (Desejado: $1.0$) — **ACERTO**
*   **Amostra 9** (0.6129, 0.4518): $y = -0.458353 \implies y_{\text{pós}} = -1.0$ (Desejado: $-1.0$) — **ACERTO**
*   **Amostra 10** (0.9948, 0.4962): $y = -0.983769 \implies y_{\text{pós}} = -1.0$ (Desejado: $-1.0$) — **ACERTO**

*   **Taxa de Acerto Final:** **80.0%**

#### Caso 2: Espalhamento via Heurística de Distância (Acurácia: 100.0%)
Ao utilizar o espalhamento de distância ($\sigma^2 = 0.065458$), os raios gaussianos cobrem melhor os limites dos centroides, corrigindo as amostras 2 e 4.
*   **Amostra 2** (0.0388, 0.2703): $y = 0.172069 \implies y_{\text{pós}} = 1.0$ — **ACERTO**
*   **Amostra 4** (0.7075, 0.1502): $y = 0.005705 \implies y_{\text{pós}} = 1.0$ — **ACERTO**
*   **Taxa de Acerto Final:** **100.0%**

---

### 5. Estratégias para Aumento de Performance
Para melhorar o desempenho de acerto em cenários mais restritivos:
1.  **Ajuste do Espalhamento ($\sigma_j$):** Realizar buscas em grade para otimizar os raios de ativação de cada Gaussiana.
2.  **Clusterização de Ambas as Classes:** Executar o K-Means considerando dados da classe sem radiação ($-1.0$) para evitar pontos cegos.
3.  **Variação de Topologia:** Aumentar o número de neurônios ocultos $M$ (ex: para 5 ou 8), o que projeta os dados em um espaço de maior dimensão facilitando a separação linear.
4.  **Regularização L2 (Tikhonov):** Prevenir overfitting de coeficientes de saída adicionando um termo penalizador na pseudo-inversa.
5.  **Ajuste Supervisionado dos Centros:** Treinar simultaneamente os centros, spreads e pesos usando gradiente descendente completo (backpropagation).

---

## Como Executar
1.  **Instalar dependências (no ambiente virtual):**
    ```bash
    pip install numpy pandas matplotlib
    ```
2.  **Treinar e gerar relatório:**
    ```bash
    python train.py
    ```
3.  **Executar testes unitários:**
    ```bash
    python test_rbf.py
    ```
