# Controllable Multi-Character Psychology-Oriented Story Generation (CIKM 2020)

In our work, we aim to design an emotional line for each character that considers multiple emotions common in psychological theories, with the goal of generating stories with richer emotional changes in the char- acters. To the best of our knowledge, this work is first to focuses on characters’ emotional lines in story generation. We present a novel model-based attention mechanism that we call SoCP (Story- telling of multi-Character Psychology). We show that the proposed model can not only generate a coherent story but also generate a story considering the changes in the psychological state of different characters. To take into account the particularity of the model, in addition to commonly used evaluation indicators(BLEU, ROUGE, etc.), we introduce the accuracy rate of psychological state control as a novel evaluation metric. The new indicator reflects the effect of the model on the psychological state control of story characters. Experiments show that with SoCP, the generated stories follow the psychological state for each character according to both automatic and human evaluations.

![image](https://user-images.githubusercontent.com/44796817/159868885-3d2b0c73-d96d-4e6f-977e-0a87122efb40.png)

## Requirements
- python3+
- nltk
- pytorch
- tqdm
- sklearn

## File Description
___train.py___ and ___test.py___ are the mainly files to train and test our model.
___train_clf.py___ and ___test_clf.py___ are the files about training and testing the emotion classifier.

## About Data
[Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/)

## Run
1. First Download the dataset [Modeling Naive Psychology of Characters in Simple Commonsense Stories](https://uwnlp.github.io/storycommonsense/) and run ___data_process.py___. Then we can get ___pro_data.csv___
2. Run ___wirte_data_and_word_dict.py___ to get ___train_pmr_idx.csv___ and ___word_dict_pmr_idx.json___
3. Run ___train.py___
