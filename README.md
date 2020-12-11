
# Evaluacion de Word Embeddings para Español


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Datasets](#datasets)
  * [Word Similarity](#word-similarity)
  * [Word Analogy](#word-analogy)
  * [Outlier Detection](#outlier-detection)
  * [Cross-match](#cross-match)
  * [Text Classification](#text-classification)
* [Usage](#usage)
* [Results](#results)
  * [Word Similarity](#word-similarity)
  * [Word Analogy](#word-analogy)
  * [Outlier Detection](#outlier-detection)
  * [Cross-match](#cross-match)
  * [Text Classification](#text-classification)
* [Reference](#reference)


<!-- ABOUT THE PROJECT -->
## About the Project

This is a tool for the validation of word embeddings for the Spanish language. This tool use different types 
of evaluation methods: Word Similarity, Word Analogy, Outlier Detection, Cross-match and Text Classification.

<!-- DATASETS -->
## Datasets

### Word Similarity

* MC-30
* RG-65
* SemEval-2017
* WordSim353
* MultiSimLex

### Word Analogy

* Google Analogy
* SATS

### Outlier Detection

* WordSim-500

### Text Classification

* 


<!-- USAGE EXAMPLE -->
## Usage

The evaluation tool is divided in two parts, the first part evaluate word embeddings using intrinsic methods.

```
python IntrinsecEvaluation.py -f <PATH> -s -a -o -c <PATH>
```

The second part evaluate word embeddings using extrinsic methods, specifically a text classification task is used.

```
python ExtrinsecEvaluation.py -f <PATH> -lstm -mean
```

It is also possible to evaluate a set of word embeddings, in those cases, the vocabulary in all datasets use is 
intersected with the vocabulary from all the word embeddings.

```
python IntrinsecEvaluation.py -d <PATH> -s -a -o -c <PATH>
```

In all the previous examples, the flags define the evaluation method to use, so it is possible to do all of them or 
just some. if necessary, you can use the -h flag to find out more information.

This tool use gensim when loading word embeddings, because of that, is possible that some word embeddings can´t be 
evaluated using this script. For those cases, there is an example code (*UseExample.ipynb*) with which you can evaluate 
these word embedding


<!-- RESULTS -->
## Results

This evaluation tool has been used to compare different word embeddings and the results obtained are presented below.
Information about the word embeddings used are shown in [Reference](#reference)

### Word Similiarity

The results shown are the mean across the different datasets used for this evaluation method.

| Word Embeddings   |  Pearson | Spearman | Kendall |
| :---------------- | :--: | :--: | :--: |
| FastText-SUC(M)   | 0.66 | 0.68 | 0.52 |
| FastText-SUC(L)   | **0.68** | **0.70** | **0.54** |
| FastText-SUC(NL)  | 0.66 | 0.69 | 0.54 |
| FastText-SBWC     | 0.64 | 0.66 | 0.50 |
| FastText-Wiki     | 0.66 | 0.69 | 0.52 |
| GloVe-SBWC        | 0.53 | 0.54 | 0.40 |
| W2V-SBWC          | 0.67 | 0.68 | 0.52 |
| BETO              | 0.44 | 0.42 | 0.29 |


### Word Analogy

The results shown are the mean across the different datasets used for this evaluation method and different types of 
analogys.

| Word Embeddings   | GA Semantic | GA Syntactic | CATS Semantic | CATS Syntactic |
| :---------------- | :--: | :--: | :--: | :--: |
| FastText-SUC(M)   |  |  |  |  |
| FastText-SUC(L)   |  |  |  |  |
| FastText-SUC(NL)  |  |  |  |  |
| FastText-SBWC     |  |  |  |  |
| FastText-Wiki     |  |  |  |  |
| GloVe-SBWC        |  |  |  |  |
| W2V-SBWC          |  |  |  |  |


### Outlier Detection

| Word Embeddings   | Accuracy | OPP |
| :---------------- | :--: | :--: |
| FastText-SUC(M)   | 0.84 | 0.63 |
| FastText-SUC(L)   | **0.86** | **0.66** |
| FastText-SUC(NL)  | 0.85 | 0.65 |
| FastText-SBWC     | 0.85 | 0.65 |
| FastText-Wiki     | 0.84 | 0.64 |
| GloVe-SBWC        | 0.82 | 0.62 |
| W2V-SBWC          | 0.78 | 0.53 |
| BETO              | 0.72 | 0.44 |


### Cross-match

| Word Embeddings | FastText-SBWC | GloVe-SBWC | W2V-SBWC |
| :-------------- | :--: | :--: | :--: |
| FastText-SBWC   | \--- | 7.81x10<sup>-28</sup> | 1.11x10<sup>-30</sup> |
| GloVe-SBWC      | 7.81x10<sup>-28</sup> | \--- | 6.7x10<sup>-28</sup> |
| W2V-SBWC        | 1.11x10<sup>-30</sup> | 6.7x10<sup>-28</sup> | \--- |


### Text Classification

The results shown correspond to task A and B of the same job where the dataset used is defined, using lstm during 
classification.

 


<!-- REFERENCE -->
## Reference

* [1] [Spanish translation of MC30]()
* [2] [Spanish translation of RG65]()
* [3] [Spanish translation of WordSim353]()
* [4] [Spanish section of SemEval-2017]()
* [5] [Spanish translation of MultiSimLex]()
* [6] [Spanish translation of Google Analogy]()
* [7] [Spanish section of WordSim500]()
* [8] [Dataset and task definition for text classification evaluation]()
