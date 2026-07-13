# Classificador de Joias

Este projeto implementa um modelo de **Image Captioning** (Legendagem de Imagens) especializado no domínio de joias. O sistema é capaz de identificar simultaneamente atributos visuais como categoria (*anel, colar, brinco*), cores (*dourado, prata*), design e pedras preciosas, gerando uma descrição textual completa baseada no template:  
`[CATEGORIA] [COR] com [DESIGN] e [PEDRA]`

A arquitetura foi inspirada no estudo de *Alcalde-Llergo et al., 2025* ([jewelry_linguistics](https://github.com/jewelryling/jewelry_linguistics)).

---

## 🚀 Como Funciona o Fluxo do Projeto

O pipeline do projeto conecta o processamento de texto e imagem em uma arquitetura híbrida **CNN (Visão Computacional) + RNN (Processamento de Linguagem Natural)**:

captions.txt + train.txt
│
▼
┌───────────────┐
│dataFunctions  │ ◄─── (Tokenização, Vocabulário e Embeddings BERTimbau)
└───────┬───────┘
▼
┌───────────────┐       ┌───────────────┐
│modelFunctions │ ◄───  │   config.py   │ (Hiperparâmetros)
│  (CNN + RNN)  │       └───────────────┘
└───────┬───────┘
▼
┌───────────────┐
│   train.py    │ ───► Gera: modelo .hdf5 + tokenizers .pkl
└───────┬───────┘
▼
┌───────────────┐
│    test.py    │ ───► Avaliação: CCR, Matriz de Confusão, Métricas e BLEU
└───────────────┘

## 🛠️ Pré-processamento e Dados

### 1. Normalização de Imagens
Antes de alimentar o modelo, as imagens físicas devem passar por uma padronização para garantir a consistência geométrica.
*   **Script:** `normalizacao_img.py`
*   **Ação:** Redimensiona as imagens para **900×900 pixels**, converte para o espaço de cores **RGB**, aplica preenchimento com **fundo branco** para manter a proporção sem distorcer a joia, e salva o resultado em formato **JPEG**.

### 2. Tratamento de Dataset Desbalanceado
Caso o seu conjunto de dados apresente uma disparidade muito grande na quantidade de imagens por categoria ou cor, o pipeline oferece duas alternativas principais:
*   **Ponderação de Perda (Class Weights):** Utiliza o arquivo `class_weights.json` gerado no split para penalizar mais severamente os erros nas classes minoritárias durante o cálculo da *Loss Function*.
*   **Data Augmentation (Opcional):** Aplicação de rotações leves, espelhamento e ajustes de brilho nas imagens físicas das classes com menos amostras antes do treino.

---

## 📁 Estrutura de Arquivos e Diretórios

```text
├── data/
│   ├── train_4096.pkl          # Features extraídas das imagens de treino (VGG16)
│   └── val_4096.pkl            # Features extraídas das imagens de validação (VGG16)
│
├── datasets/
│   ├── captions.txt            # Mapeamento global: imagem ──> legenda/classe
│   ├── train.txt               # Lista com os nomes das imagens de treino
│   ├── validation.txt          # Lista com os nomes das imagens de validação
│   ├── test.txt                # Lista com os nomes das imagens de teste
│   ├── class_weights.json      # Pesos calculados para compensar o desbalanceamento
│   ├── train/                  # Diretório com as imagens físicas de treino
│   ├── validation/             # Diretório com as imagens físicas de validação
│   └── test/                   # Diretório com as imagens físicas de teste
│
└── src/
    ├── config.py               # Configurações globais e hiperparâmetros
    ├── dataFunctions.py        # Funções para manipulação e tokenização de dados
    ├── modelFunctions.py       # Definição e arquitetura da rede (CNN + RNN)
    ├── train.py                # Script principal de treinamento do modelo
    └── test.py                 # Script de avaliação e geração de métricas
