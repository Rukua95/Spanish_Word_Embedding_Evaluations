
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
More about the dataset used are shown in [Reference](#reference).

<!-- DATASETS -->
## Datasets

### Word Similarity

* MC-30 \(translation\)
* RG-65 \(translation\)
* SemEval-2017 \(spanish section\)
* WordSim353 \(translation\)
* MultiSimLex \(spanish section\)

### Word Analogy

* Google Analogy \(translation\)
* SATS

### Outlier Detection

* WordSim-500 \(spanish section\)

### Text Classification

* ArgumentMining2017


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
More about the word embeddings used are shown in [Reference](#reference)

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

The results shown correspond to task A and B defined in the same job where the dataset used is shown. Results 
were obtained using an LSTM network in the classification process.



<!-- REFERENCE -->
## Reference

* [1] [About spanish translation of MC30](https://www.aclweb.org/anthology/D09-1124/)
* [2] [About spanish translation of RG65](https://www.aclweb.org/anthology/P15-2001/)
* [3] [About spanish translation of WordSim353](https://www.aclweb.org/anthology/D09-1124/)
* [4] [About spanish section of SemEval-2017](https://www.aclweb.org/anthology/S17-2002/)
* [5] [About spanish section of MultiSimLex](https://multisimlex.com/)
* [6] [About spanish translation of Google Analogy](https://crscardellino.github.io/SBWCE/)
* [7] [About spanish section of WordSim500](https://www.semanticscholar.org/paper/Automated-Generation-of-Multilingual-Clusters-for-Blair-Merhav/ba14d02895ed8d93b3d44e5451be83f3c9e767fa)
* [8] [About Outlier Detection task](https://www.aclweb.org/anthology/W16-2508/)
* [8] [About Cross-match task](https://www.aclweb.org/anthology/W17-5303/)
* [10] [Dataset for text classification evaluation](https://github.com/uchile-nlp/ArgumentMining2017)
* [11] [Task definition for text classification evaluation](https://www.aclweb.org/anthology/W17-5101/)
