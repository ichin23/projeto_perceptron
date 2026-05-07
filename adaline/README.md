# Projeto ADALINE - Resultados e Análises

**🔗 Link para o Relatório Interativo:** [https://projeto-perceptron.vercel.app/adaline/report.html](https://projeto-perceptron.vercel.app/adaline/report.html)

## 📌 Sobre o Projeto
Este projeto consiste na implementação da rede neural ADALINE (Adaptive Linear Neuron). Diferente do Perceptron que foca exclusivamente na separação de classes via função degrau no aprendizado, o ADALINE aplica a Regra de Widrow-Hoff (Regra Delta) para minimizar o Erro Quadrático Médio (MSE) ao longo do treinamento antes de passar a saída por uma função de ativação para classificação. O projeto simula a recepção de sinais de válvulas industriais com ruído de transmissão e foca em encontrar os pesos ideais que garantam o menor erro possível.

*Este trabalho foi desenvolvido para a disciplina de Inteligência Artificial do curso de Sistemas de Informação do CEFET-MG Varginha (1º semestre de 2026), com o Prof. Dr. Lazaro Eduardo.*

---

## 1. Desempenho do Treinamento

Foram executados 5 treinamentos para a rede ADALINE, inicializando o vetor de pesos em cada treinamento com valores aleatórios entre 0 e 1, usando uma taxa de aprendizado (η) de 0.0025 e critério de parada (ε) de 10⁻⁶.

| Treino | w0_ini | w1_ini | w2_ini | w3_ini | w4_ini | w0_fin | w1_fin | w2_fin | w3_fin | w4_fin | Épocas | Acurácia |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **T1** | 0.5488 | 0.7152 | 0.6028 | 0.5449 | 0.4237 | 1.8112 | 1.3125 | 1.6413 | -0.4264 | -1.1771 | 831 | 91.4% |
| **T2** | 0.3745 | 0.9507 | 0.7320 | 0.5987 | 0.1560 | 1.8112 | 1.3126 | 1.6414 | -0.4263 | -1.1771 | 832 | 91.4% |
| **T3** | 0.9439 | 0.0885 | 0.6954 | 0.7000 | 0.4996 | 1.8113 | 1.3125 | 1.6413 | -0.4265 | -1.1771 | 817 | 91.4% |
| **T4** | 0.0458 | 0.5861 | 0.2032 | 0.0842 | 0.0260 | 1.8113 | 1.3125 | 1.6413 | -0.4266 | -1.1771 | 848 | 91.4% |
| **T5** | 0.8034 | 0.5275 | 0.1191 | 0.6397 | 0.0909 | 1.8113 | 1.3125 | 1.6413 | -0.4265 | -1.1771 | 829 | 91.4% |

---

## 2. Resultados de Classificação de Teste

Para todos os treinamentos realizados acima, a rede ADALINE foi aplicada para classificar as seguintes amostras de teste, decidindo se os sinais seriam encaminhados para a Válvula A (-1) ou B (1):

| Amostra | x1 | x2 | x3 | x4 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.9694 | 0.6909 | 0.4334 | 3.4965 | -1 | -1 | -1 | -1 | -1 |
| **2** | 0.5427 | 1.3832 | 0.6390 | 4.0352 | -1 | -1 | -1 | -1 | -1 |
| **3** | 0.6081 | -0.9196 | 0.5925 | 0.1016 | 1 | 1 | 1 | 1 | 1 |
| **4** | -0.1618 | 0.4694 | 0.2030 | 3.0117 | -1 | -1 | -1 | -1 | -1 |
| **5** | 0.1870 | -0.2578 | 0.6124 | 1.7749 | -1 | -1 | -1 | -1 | -1 |
| **6** | 0.4891 | -0.5276 | 0.4378 | 0.6439 | 1 | 1 | 1 | 1 | 1 |
| **7** | 0.3777 | 2.0149 | 0.7423 | 3.3932 | 1 | 1 | 1 | 1 | 1 |
| **8** | 1.1498 | -0.4067 | 0.2469 | 1.5866 | 1 | 1 | 1 | 1 | 1 |
| **9** | 0.9325 | 1.0950 | 1.0359 | 3.3591 | 1 | 1 | 1 | 1 | 1 |
| **10** | 0.5060 | 1.3317 | 0.9222 | 3.7174 | -1 | -1 | -1 | -1 | -1 |
| **11** | 0.0497 | -2.0656 | 0.6124 | -0.6585 | -1 | -1 | -1 | -1 | -1 |
| **12** | 0.4004 | 3.5369 | 0.9766 | 5.3532 | 1 | 1 | 1 | 1 | 1 |
| **13** | -0.1874 | 1.3343 | 0.5374 | 3.2189 | -1 | -1 | -1 | -1 | -1 |
| **14** | 0.5060 | 1.3317 | 0.9222 | 3.7174 | -1 | -1 | -1 | -1 | -1 |
| **15** | 1.6375 | -0.7911 | 0.7537 | 0.5515 | 1 | 1 | 1 | 1 | 1 |

*(Veja os gráficos de Erro Quadrático Médio em função da época de treinamento no [Relatório Completo](https://projeto-perceptron.vercel.app/adaline/report.html))*

---

## 3. Análise das Perguntas

### Embora o número de épocas de cada treinamento realizado no item 2 seja diferente, explique por que então os valores dos pesos continuam praticamente inalterados.

A rede ADALINE utiliza a regra de aprendizado de Widrow-Hoff (regra Delta), que busca minimizar a função de custo do Erro Quadrático Médio (MSE). No caso da ADALINE, como a função de ativação durante o processo de treinamento é linear, a superfície de erro apresenta-se como uma **paraboloide multidimensional convexa**.

Isso significa que a função de erro possui um **único mínimo global**, sem mínimos locais que possam "prender" o algoritmo. Independentemente de onde os pesos sejam inicializados (como evidenciado pelas sementes aleatórias dos 5 treinamentos), o algoritmo de gradiente descendente sempre descerá a superfície em direção a esse exato mesmo ponto ótimo.

A variação no número de épocas ocorre estritamente porque cada sorteio inicial "solta" o modelo em distâncias topográficas diferentes desse mínimo global. Alguns sorteios iniciam o treinamento mais distantes, exigindo mais passos (épocas) para que a variação do erro (ΔMSE) atinja a precisão limite (ε = 10⁻⁶), mas o vetor de pesos resultante no final será sempre, virtualmente, idêntico, traduzindo-se na solução global perfeita para aquele dataset específico.
