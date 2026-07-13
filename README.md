# Classificador de Joias

Este projeto implementa um modelo de **Image Captioning** (Legendagem de Imagens) especializado no domínio de joias. O sistema é capaz de identificar simultaneamente atributos visuais como categoria (*anel, colar, brinco*), cores (*dourado, prata*), design e pedras preciosas, gerando uma descrição textual completa baseada no template:  `[CATEGORIA] [COR] com [DESIGN] e [PEDRA]`

A arquitetura foi inspirada no estudo de *Alcalde-Llergo et al., 2025* ([jewelry_linguistics](https://github.com/jewelryling/jewelry_linguistics)).

---

## 🚀 Como Funciona o Fluxo do Projeto

O pipeline do projeto conecta o processamento de texto e imagem em uma arquitetura híbrida **CNN (Visão Computacional) + RNN (Processamento de Linguagem Natural)**:

```mermaid
graph TD
    %% Definição dos nós
    A["📄 captions.txt + train.txt"] --> B["⚙️ dataFunctions.py<br>(Tokenização, Vocabulário e Embeddings BERTimbau)"]
    
    C["⚙️ config.py<br>(Hiperparâmetros)"] --> D
    B --> D["🧠 modelFunctions.py<br>(CNN + RNN)"]
    
    D --> E["🏋️‍♂️ train.py<br><b>Gera:</b> modelo .hdf5 + tokenizers .pkl"]
    E --> F["🧪 test.py<br><b>Avaliação:</b> CCR, Matriz de Confusão, Métricas e BLEU"]

    %% Estilização para ficar elegante
    style A fill:#f9f9f9,stroke:#ddd,stroke-width:1px,color:#333
    style B fill:#e1f5fe,stroke:#0288d1,stroke-width:1px,color:#01579b
    style C fill:#f9f9f9,stroke:#ddd,stroke-width:1px,color:#333
    style D fill:#e8f5e9,stroke:#388e3c,stroke-width:1px,color:#1b5e20
    style E fill:#fff3e0,stroke:#f57c00,stroke-width:1px,color:#e65100
    style F fill:#ede7f6,stroke:#7e57c2,stroke-width:1px,color:#4a148c
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

## 📝 Descrição dos Módulos (`src/`)

### ⚙️ `config.py`
Centraliza as constantes de configuração compartilhadas por todo o ecossistema do projeto:
*   **Hiperparâmetros:** `LEARNING_RATE`, `EMBEDDING_SIZE`, `OPTIMIZER` (Adam).
*   **NLP:** `EMBEDDING_NAME` (Uso do modelo pré-treinado BERTimbau), tokens especiais `<START>` e `<END>`.
*   **Configurações do Modelo:** Dimensões de entrada de imagem padrão da VGG16 (224×224), funções de perda (`LOSS`) e métricas de monitoramento.

### 📊 `dataFunctions.py`
Responsável por estruturar a entrada de dados textuais.
*   `getData()`: Lê o arquivo `captions.txt` e retorna um dicionário estruturado `{imagem: legenda}`.
*   `getLexicon()`: Filtra e extrai o vocabulário único de todas as legendas.
*   `getDataArrays()`: Cruza os dicionários com as listas de splits (`train.txt`/`val.txt`).
*   `getTokenizers()`: Constrói os mapeamentos numéricos index-para-palavra (`idxtoword`) e palavra-para-index (`wordtoidx`).
*   `getTokensArrays()`: Transforma as legendas de texto puro em sequências numéricas de índices.
*   `getBERTimbauEmbeddings()`: Conecta ao modelo linguístico para extrair vetores semânticos de 768 dimensões para cada palavra do vocabulário.
*   `getEmbeddingMatrix()`: Estrutura a matriz final de pesos de embedding que será injetada na camada inicial da rede recorrente.

### 🧠 `modelFunctions.py`
Gerencia a arquitetura integrada do modelo.
*   **Módulo Vision (CNN):** Suporta o carregamento de backbones pré-treinados como VGG16, Inception ou MobileNet. A função `encode_image()` extrai o vetor de características denso de 4096 posições.
*   **Módulo Linguístico (RNN):** Constrói a rede recorrente usando células GRU (`build_gru_model()`) ou LSTM (`build_lstm_model()`).
*   `compile_model()`: Acopla a rede à matriz de embeddings do BERTimbau.
*   `create_generator()`: Pipeline de dados customizado que entrega lotes (*batches*) de dados com embaralhamento ativo a cada nova época de treino.
*   `generate_caption()`: O motor de inferência que recebe uma imagem e gera a legenda preditiva palavra por palavra até encontrar o token de encerramento.

### 🏋️‍♂️ `train.py`
Executa o fluxo completo de aprendizado da rede:
1. Carrega as referências de dados estruturados.
2. Extrai e gera o cache das características visuais na pasta `data/*.pkl` (via técnica de *Transfer Learning*).
3. Constrói o vocabulário e baixa os pesos semânticos do BERTimbau.
4. Inicializa o treinamento aplicando callbacks de resiliência: `EarlyStopping` (interrompe o treino se o modelo parar de evoluir) e `ReduceLROnPlateau` (reduz a taxa de aprendizado ao encontrar um platô de perda).
5. Exporta o modelo final treinado no formato `.hdf5` e os tokenizadores correspondentes.

### 🧪 `test.py`
Avalia rigorosamente o desempenho do modelo no conjunto de testes (`test/`):
1. Faz a inferência de novas imagens usando o método autoregressivo `generate_caption()`.
2. Avalia a exatidão através da métrica **CCR (Correct Classification Rate)** para verificar se a legenda gerada é 100% idêntica à esperada.
3. Mede a acurácia isolada por subcategoria (anel, brinco, colar).
4. Plota matrizes de confusão e calcula métricas clássicas de classificação: Precisão, Recall e F1-Score.
5. Calcula a métrica **BLEU**, padrão internacional para avaliar a qualidade de textos gerados artificialmente.
6. Consolida e exporta todos os relatórios estruturados em arquivos CSV dentro de `models/test_logs/`.
