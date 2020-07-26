# Controllable Multi-Character Psychology-Oriented Story Generation (CIKM 2020)

In our work, we aim to design an emotional line for each character that considers multiple emotions common in psychological theories, with the goal of generating stories with richer emotional changes in the char- acters. To the best of our knowledge, this work is first to focuses on characters’ emotional lines in story generation. We present a novel model-based attention mechanism that we call SoCP (Story- telling of multi-Character Psychology). We show that the proposed model can not only generate a coherent story but also generate a story considering the changes in the psychological state of different characters. To take into account the particularity of the model, in addition to commonly used evaluation indicators(BLEU, ROUGE, etc.), we introduce the accuracy rate of psychological state control as a novel evaluation metric. The new indicator reflects the effect of the model on the psychological state control of story characters. Experiments show that with SoCP, the generated stories follow the psychological state for each character according to both automatic and human evaluations.

## Requirements
- python3+
- nltk
- pytorch

## About Data
[Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/)
