# classificador-joias

Esse projeto contém um modelo classificador de atributos de joias que identifica categorias como: anel, colar e brinco. Cores como: dourado, prata. E explora a geração de legendas completa de joias considerando um template [CATEGORIA] [COR] com [DESIGN] e [PEDRA]. 

-------------------------------------------------------

Pré processamento 

****Com todas imagens baixadas, deve-se aplicar a seguinte normalização:

normalizacao_img.py - redimensiona as imagens para 900×900 pixels com preenchimento em fundo branco, converte para RGB e salva em JPEG.

****Caso haja um dataset desbalanceado, uma possível alternativa:

-------------------------------------------------------

Dataset:

Pasta data - Armazena as imagens encodadas pelo VGG16 em formato .pk1 (pickle):

Cada imagem é codificada uma única vez pelo VGG16 e salva em disco. Nas execuções seguintes o pipeline carrega direto do .pk1 em vez de reprocessar todas as imagens — isso reduz significativamente o tempo de treino.

-train_4096.pk1 ← features das imagens de treino 
-val_4096.pk1  ← features das imagens de validação

Pasta datasets: Onde deve conter os datasets organizados com suas estruturas de split:

├── dataset/
│   ├── captions.txt      ← imagem → classe
│   ├── train.txt         ← lista de imagens de treino
│   ├── validation.txt
│   ├── test.txt
│   ├── class_weights.json ← pesos por classe para compensar desbalanceamento
│   ├── train/            ← imagens físicas de treino
│   ├── validation/       ← imagens físicas de validação
│   └── test/             ← imagens físicas de teste

-------------------------------------------------------

A arquitetura de treino foi inspirado no estudo de Alcalde-Llergo et al., 2025 (https://github.com/jewelryling/jewelry_linguistics)

Fluxo: 
 
captions.txt + train.txt
       ↓
  dataFunctions.py
  (tokeniza, embeds)
       ↓
  modelFunctions.py        config.py
  (CNN + RNN)              (hiperparâmetros)
       ↓
    train.py → modelo .hdf5 + tokenizers .pk1
       ↓
    test.py  → CCR, matriz de confusão, métricas


-------------------------------------------------------

Arquivos do projeto — src/

config.py — configurações globais

Define constantes usadas por todos os outros arquivos:
LEARNING_RATE, EMBEDDING_SIZE, EMBEDDING_NAME (BERTimbau), tokens START/END, dimensões de entrada da VGG16 (224×224), OPTIMIZER(Adam), LOSS(Função de perda), METRICS

-------------------------------------------------------

dataFunctions.py — funções de dados

getData()          → lê captions.txt → dicionário {imagem: legenda}
getLexicon()       → extrai vocabulário único das legendas
getDataArrays()    → lê train.txt/val.txt → listas de imagens e legendas
getTokenizers()    → cria idxtoword e wordtoidx (índice ↔ palavra)
getTokensArrays()  → converte legendas em sequências de índices numéricos
getBERTimbauEmbeddings() → gera vetores 768-dim para cada palavra
getEmbeddingMatrix()     → monta matriz de embeddings para o modelo

-------------------------------------------------------

modelFunctions.py — arquitetura do modelo CNN+RNN

CNNModel  → carrega VGG16/Inception/MobileNet pré-treinado
            encode_image() extrai features (vetor 4096-dim)

RNNModel  → arquitetura GRU ou LSTM
            build_gru_model()    → monta rede GRU
            build_lstm_model()   → monta rede LSTM
            compile_model()      → aplica embeddings BERTimbau
            create_generator()   → gera batches com shuffle a cada época
            generate_caption()   → gera legenda token a token para uma imagem

-------------------------------------------------------

train.py — treinamento

1. Lê datasets (captions.txt, train.txt, val.txt)
2. Embaralha dados de treino
3. Codifica imagens via CNN → salva em data/*.pk1
4. Gera embeddings BERTimbau
5. Carrega pesos do modelo pré-treinado (transfer learning)
6. Treina com EarlyStopping e ReduceLROnPlateau
7. Salva modelo (.hdf5) e tokenizers (idxtoword/wordtoidx .pk1)

-------------------------------------------------------

test.py — avaliação do modelo

1. Carrega modelo treinado
2. Para cada imagem do test/:
   → gera legenda com generate_caption()
   → compara com legenda esperada
3. Calcula CCR exato (legenda gerada == legenda original)
4. Calcula acurácia por categoria (anel/brinco/colar)
5. Gera matriz de confusão e métricas (recall, precisão, F1) e BLEU para tarefa de gerar legendas.
6. Salva CSV de métricas em models/test_logs/

-------------------------------------------------------

 
 
