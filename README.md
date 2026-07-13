# Classificador de Joias

Este projeto implementa um modelo de **Image Captioning** (Legendagem de Imagens) especializado no domínio de joias. O sistema é capaz de identificar simultaneamente atributos visuais como categoria (*anel, colar, brinco*), cores (*dourado, prata*), design e pedras preciosas, gerando uma descrição textual completa baseada no template:  
`[CATEGORIA] [COR] com [DESIGN] e [PEDRA]`

A arquitetura foi inspirada no estudo de *Alcalde-Llergo et al., 2025* ([jewelry_linguistics](https://github.com/jewelryling/jewelry_linguistics)).

---

## 🚀 Como Funciona o Fluxo do Projeto

O pipeline do projeto conecta o processamento de texto e imagem em uma arquitetura híbrida **CNN (Visão Computacional) + RNN (Processamento de Linguagem Natural)**:
