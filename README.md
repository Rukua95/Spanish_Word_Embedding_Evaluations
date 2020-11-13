# Evaluation of Spanish Word Embeddings


<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Evaluations and Datasets](#evaluation-and-datasets)
  * [Word Similarity](#word-similarity)
  * [Word Analogy](#word-analogy)
  * [Outlier Detection](#outlier-detection)
  * [Cross-match](#cross-match)
  * [Text Classification](#text-classification)
* [Usage](#usage)
* [Results](#results)
  * [Word Similarity](#word-similarity-results)
  * [Word Analogy](#word-similarity-results)
  * [Outlier Detection](#word-similarity-results)
  * [Cross-match](#word-similarity-results)
  * [Text Classification](#word-similarity-results)
* [Reference](#reference)



<!-- ABOUT THE PROJECT -->
## About The Project

This is a tool for the validation of word embeddings for the Spanish language. This tool use differents types 
of evaluation methods: Word Similarity, Word Analogy, Outlier Detection, Cross-match and Text Classification.

<!-- EVALUATION AND DATASETS -->
## Evaluations and Datasets

### Word Similiarity



### Word Analogy

### Outlier Detection

### Cross-match

### Text Classification


<!-- USAGE EXAMPLE -->
## Usage



```
python IntrinsecEvaluation.py -f <PATH> -s -a -o -c <PATH>
```

```
python ExtrinsecEvaluation.py -f <PATH> -lstm -mean
```

```

```


<!-- RESULTS -->
## Results

This evaluation tool has been used to compare differents word embeddings. The results obtained are presented below as 
the mean results of every dataset.

### Word Similiarity

| Word Embeddings   |  Pearson | Spearman | Kendall |
| :---------------- | :--: | :--: | :--: |
| Fasttext-SUC(M)   | 0.66 | 0.68 | 0.52 |
| Fasttext-SUC(L)   | 0.68 | 0.70 | 0.54 |
| Fasttext-SUC(NL)  | asdf | asdf | asdf |
| Fasttext-SBWC     | asdf | asdf | asdf |
| Fasttext-Wiki     | asdf | asdf | asdf |
| GloVe-SBWC        | asdf | asdf | asdf |
| W2V-SBWC          | asdf | asdf | asdf |
| BETO              | asdf | asdf | asdf |


### Word Analogy

| Word Embeddings   | GA Semantic | GA Sintactic | CATS Semantic | CATS Sintactic |
| :---------------- | :--: | :--: | :--: | :--: |
| Fasttext-SUC(M)   | 0.66 | 0.68 | 0.52 | asdf |
| Fasttext-SUC(L)   | 0.68 | 0.70 | 0.54 | asdf |
| Fasttext-SUC(NL)  | asdf | asdf | asdf | asdf |
| Fasttext-SBWC     | asdf | asdf | asdf | asdf |
| Fasttext-Wiki     | asdf | asdf | asdf | asdf |
| GloVe-SBWC        | asdf | asdf | asdf | asdf |
| W2V-SBWC          | asdf | asdf | asdf | asdf |


### Outlier Detection

| Word Embeddings   | Accuracy | OPP |
| :---------------- | :--: | :--: |
| Fasttext-SUC(M)   | 0.84 | 0.63 |
| Fasttext-SUC(L)   | **0.86** | **0.66** |
| Fasttext-SUC(NL)  | 0.85 | 0.65 |
| Fasttext-SBWC     | 0.85 | 0.65 |
| Fasttext-Wiki     | 0.84 | 0.64 |
| GloVe-SBWC        | 0.82 | 0.62 |
| W2V-SBWC          | 0.78 | 0.53 |
| BETO              | 0.72 | 0.44 |


### Cross-match

| Word Embeddings | Fasttext-SBWC | GloVe-SBWC | W2V-SBWC |
| :-------------- | :--: | :--: | :--: |
| Fasttext-SBWC   | \--- | 7.81x10<sup>-28</sup> | 1.11x10<sup>-30</sup> |
| GloVe-SBWC      | 7.81x10<sup>-28</sup> | \--- | 6.7x10<sup>-28</sup> |
| W2V-SBWC        | 1.11x10<sup>-30</sup> | 6.7x10<sup>-28</sup> | \--- |


### Text Classification



<!-- REFERENCE -->
## Reference