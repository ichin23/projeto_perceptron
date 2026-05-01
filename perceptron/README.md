# Projeto Perceptron - Resultados e Análises

**🔗 Link para o Dashboard em Produção:** [https://projeto-perceptron.vercel.app/perceptron/report.html](https://projeto-perceptron.vercel.app/perceptron/report.html)

## 📌 Sobre o Projeto
Este projeto consiste na implementação e análise do Perceptron, o bloco fundamental das redes neurais artificiais. O objetivo principal é aplicar o algoritmo de aprendizado (Regra de Hebb Supervisionada) para classificar amostras de óleo de forma automática. O trabalho abrange a inicialização aleatória de pesos, o registro de múltiplas iterações (épocas) até a convergência e uma reflexão teórica sobre o comportamento do modelo diante de problemas de classificação.

*Este trabalho foi desenvolvido para a disciplina de Inteligência Artificial do curso de Sistemas de Informação do CEFET-MG Varginha (1º semestre de 2026), com o Prof. Dr. Lazaro Eduardo.*

---

## 1. Desempenho do Treinamento

Foram executados 5 treinamentos para a rede perceptron, inicializando o vetor de pesos em cada treinamento com valores aleatórios entre 0 e 1.

| Treinamento | Inicial w0 | Inicial w1 | Inicial w2 | Inicial w3 | Final w0 | Final w1 | Final w2 | Final w3 | Número de Épocas |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1º (T1)** | 0.6115 | 0.9784 | 0.8572 | 0.9541 | -1.6085 | 1.1068 | 2.1997 | -0.4537 | 173 |
| **2º (T2)** | 0.2267 | 0.0337 | 0.4380 | 0.8580 | -1.4733 | 1.0469 | 2.1170 | -0.4315 | 156 |
| **3º (T3)** | 0.9045 | 0.9463 | 0.4174 | 0.2608 | -1.5355 | 1.0804 | 2.1602 | -0.4460 | 166 |
| **4º (T4)** | 0.7952 | 0.7677 | 0.3386 | 0.1416 | -1.5448 | 1.0906 | 2.1845 | -0.4442 | 174 |
| **5º (T5)** | 0.9728 | 0.4058 | 0.0511 | 0.5994 | -1.3272 | 0.9877 | 2.0977 | -0.4068 | 153 |

---

## 2. Resultados de Classificação

Após o treinamento do perceptron, o mesmo foi aplicado na classificação automática das seguintes amostras de óleo. Os resultados das saídas (Classes) referentes aos cinco processos de treinamento realizados estão descritos na tabela abaixo:

| Amostra | x1 | x2 | x3 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| **2** | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| **3** | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| **4** | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| **5** | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| **6** | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| **7** | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| **8** | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| **9** | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| **10** | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

---

## 3. Análise das Perguntas

### Explique por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron.
A variabilidade observada no número de épocas (de 153 a 174) é um reflexo direto da estocasticidade na inicialização. No espaço de alta dimensionalidade dos pesos, a "paisagem" de erro contém múltiplas trajetórias para a convergência. Ao sortear pesos entre 0 e 1, posicionamos o modelo em diferentes "regiões de largada". Algumas dessas regiões possuem um gradiente de erro mais favorável ou estão geometricamente mais próximas de uma fronteira de decisão válida, permitindo que a Regra de Hebb Supervisionada ajuste o hiperplano em menos iterações.

### Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões.
O Perceptron é o bloco fundamental das redes neurais, mas sua arquitetura de camada única impõe uma restrição geométrica severa: a **Separabilidade Linear**. Ele assume que o mundo pode ser dividido por uma régua reta (ou hiperplano). Em cenários reais, onde as propriedades podem interagir de forma não-linear, o Perceptron simples falha, pois não consegue criar curvas ou fronteiras complexas de separação. Para superar isso, seriam necessárias camadas ocultas (MLP - Multi-Layer Perceptron) com funções de ativação não-lineares, permitindo a criação de superfícies de decisão complexas.
