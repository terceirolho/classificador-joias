# Projeto Joias

Este projeto implementa um modelo de **Image Captioning** (Legendagem de Imagens) especializado no domínio de joias. Este projeto implementa um modelo de Image Captioning especializado em rótulos disponiveis em joias de uma marca brasileiras, desenvolvido como parte do TCC de MBA. O modelo foi treinado com o intuito de ser capaz descrever atributos visuais de joias a partir de imagens, gerando legendas estruturadas no formato: [CATEGORIA] [COR] com design [DESIGN]. ou [CATEGORIA] [COR] com design [DESIGN] e com [PEDRA].

Foi usado transfer learning em cascata entre três tarefas.
Os atributos reconhecidos são:

Categoria: Anel, Brinco, Colar — acurácia de 87,35% no teste
Cor: Dourado, Prata, Dourado e Prata — acurácia de 89,1% no teste
Design: Orgânico, Escultural, Minimalista, Figurativo, Geométrico, Letter, Maximalista — acurácia de 26,5% no teste
Pedra: 15 tipos incluindo Pérola, Zircônia, Quartzo e Pedra Natural — acurácia de 0% no teste

A arquitetura foi inspirada no estudo de *Alcalde-Llergo et al., 2025* ([jewelry_linguistics](https://github.com/jewelryling/jewelry_linguistics)). 


---

## 🚀 Como Funciona o Fluxo do Projeto

O pipeline do projeto conecta o processamento de texto e imagem em uma arquitetura híbrida **CNN (Visão Computacional) + RNN (Processamento de Linguagem Natural)**:

````mermaid
flowchart TD
    A["Pré-processamento\nNormalização · Splits · Captions"] --> B["dataFunctions.py\nVocabulário e Tokenização"]
    B --> D["modelFunctions.py\nVGG16 + GRU"]
    C["config.py"] --> D
    D --> E["train.py\nTransfer Learning em Cascata\n Dataset [IDENTIFICAR CATEGORIA] → Dataset [IDENTIFICAR COR] → Dataset [GERAR LEGENDAS]"]
    E --> F["Modelos salvos\n.hdf5 · .pk1"]
    F --> G["test.py\nInferência e Avaliação"]
    G --> H["Métricas\nAcurácia · F1 · CCR · BLEU"]
````


## 🛠️ Pré-processamento e Dados - Foi complexo e precisou de uma análise profunda

### 1. Normalização de Imagens
Antes de alimentar o modelo, as imagens físicas devem passar por uma padronização e ser usado no treinamento do modelo.
*   **Script:** `normalizacao_img.py`
*   **Ação:** Redimensiona as imagens para **900×900 pixels**, converte para o espaço de cores **RGB**, aplica preenchimento com **fundo branco** para manter a proporção sem distorcer a joia, e salva o resultado em formato **JPEG**.

### 2. Tratamento de Dataset Desbalanceado
Caso o seu conjunto de dados apresente uma disparidade muito grande na quantidade de imagens por classe, o pipeline oferece como alternativa o uso de Class Weights no treino:
*   **Ponderação de Perda** Utiliza o arquivo `class_weights.json` gerado no split para penalizar mais severamente os erros nas classes minoritárias durante o cálculo da *Loss Function*.
 

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
*   **NLP:** `EMBEDDING_NAME` (Embora há presença do BERTimbau, não foi usado pois foi feito um experimento anterior e visto que não obteve melhoras), tokens especiais `<startcap>` e `<endcap>`.
*   **Configurações do Modelo:** Dimensões de entrada de imagem padrão da VGG16 (224×224), funções de perda (`LOSS`) e métricas de monitoramento.

### 📊 `dataFunctions.py`
Responsável por estruturar a entrada de dados textuais.
*   `getData()`: Lê o arquivo `captions.txt` e retorna um dicionário estruturado `{imagem: legenda}`.
*   `getLexicon()`: Filtra e extrai o vocabulário único de todas as legendas.
*   `getDataArrays()`: Cruza os dicionários com as listas de splits (`train.txt`/`val.txt`).
*   `getTokenizers()`: Constrói os mapeamentos numéricos index-para-palavra (`idxtoword`) e palavra-para-index (`wordtoidx`).
*   `getTokensArrays()`: Transforma as legendas de texto puro em sequências numéricas de índices.
*   `getBERTimbauEmbeddings()`: Conecta ao modelo linguístico para extrair vetores semânticos de 768 dimensões para cada palavra do vocabulário. **Mas não foi usado 
*   `getEmbeddingMatrix()`: Estrutura a matriz final de pesos de embedding que será injetada na camada inicial da rede recorrente.

### 🧠 `modelFunctions.py`
Gerencia a arquitetura integrada do modelo.
*   **Módulo Vision (CNN):** Suporta o carregamento de backbones pré-treinados como VGG16, Inception ou MobileNet. A função `encode_image()` extrai o vetor de características denso de 4096 posições.
*   **Módulo Linguístico (RNN):** Constrói a rede recorrente usando células GRU (`build_gru_model()`) *Há LSTM, mas não foi usada (`build_lstm_model()`).
*   `compile_model()`: Compila o modelo com o otimizador e a função de perda.
*   `create_generator()`: Pipeline de dados customizado que entrega lotes (*batches*) de dados com embaralhamento ativo a cada nova época de treino.
*   `generate_caption()`: Recebe imagem e gera a legenda preditiva palavra por palavra até encontrar o token de encerramento.

### 🏋️‍♂️ `train.py`
Executa o fluxo completo de aprendizado da rede:
1. Carrega as referências de dados estruturados.
2. Extrai e gera o cache das características visuais na pasta `data/*.pk1` (via técnica de *Transfer Learning*).
3. Constrói o vocabulário a partir dos captions de cada dataset. *** BERTimbau não foi usado.
4. Aplica class weights carregados do class_weights.json para compensar o desbalanceamento entre classes.
5. Inicializa o treinamento aplicando callbacks de resiliência: `EarlyStopping` (interrompe o treino se o modelo parar de evoluir) e `ReduceLROnPlateau` (reduz a taxa de aprendizado ao encontrar um platô de perda).
6. Entrega o modelo final treinado no formato `.hdf5` e os tokenizadores correspondentes.

### 🧪 `test.py`
Avalia rigorosamente o desempenho do modelo no conjunto de testes (`test/`):
1. Faz a inferência de novas imagens usando o método autoregressivo `generate_caption()`.
2. Avalia a exatidão através da métrica **CCR (Correct Classification Rate)** para verificar se a legenda gerada é 100% idêntica à esperada.
3. Mede a acurácia isolada por subcategoria (anel, brinco, colar).
4. Plota matrizes de confusão e calcula métricas clássicas de classificação: Precisão, Recall e F1-Score.
5. Calcula a métrica **BLEU**, padrão internacional para avaliar a qualidade de textos gerados. **Apenas para a TAREFA 3 [GERAR LEGENDAS]
6. Consolida e exporta todos os relatórios estruturados em arquivos CSV dentro de `models/test_logs/` em especial o test da tarefa de gerar legendas.
 
