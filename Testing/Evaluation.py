#!/usr/bin/env python

import os
import io
import sys
import argparse

from gensim.models.keyedvectors import KeyedVectors

from SimilarityTest import SimilarityTestClass
from AnalogyTest import AnalogyTestClass
from OutlierDetectionTest import OutlierDetectionTestClass
from CrossMatchTest import CrossMatchTestClass


class Embedding:
    def __init__(self, embedding):
        self._embedding = embedding

    def __getitem__(self, item):
        return self._embedding[item]

    def __contains__(self, item):
        return (item in self._embedding)

    # Entrega tamaño de los vectores dentro del embedding
    def vectorSize(self):
        return self._embedding.vector_size

    # Entrega lista de todas las palabras en el vocabulario del embedding.
    # Las palabras estan en el mismo orden que la lista de vectores
    def getWordList(self):
        return self._embedding.index2word

    # Entrega lista de todos los vectores en el embedding.
    # Los vectores estan en el mismo orden que la lista de palabras
    def getVectors(self):
        self._embedding.init_sims()
        return self._embedding.vectors_norm


def getWordEmbedding(embedding_path):
    wordvector_file = embedding_path

    return KeyedVectors.load_word2vec_format(wordvector_file, limit=None)

def main():
    #print(sys.argv)

    my_parser = argparse.ArgumentParser(prog='evaluation',
                                        description='Word embedding evaluation program.')

    # Add the arguments
    my_parser.add_argument('path',
                           metavar='PATH',
                           type=str,
                           help='Path to word embedding.')

    my_parser.add_argument('-a',
                           action='store',
                           metavar='DATASET',
                           type=str,
                           nargs='*',
                           dest='analogy_datasets',
                           help='Evaluate using word analogy. If no dataset is given, the evaluation use all dataset avalible. By default 3CosMul is used during evaluation.')

    my_parser.add_argument('-s',
                           action='store',
                           metavar='DATASET',
                           type=str,
                           nargs='*',
                           dest='similarity_datasets',
                           help='Evaluate using semantic similarity. If no dataset is given, the evaluation use all dataset avalible.')

    my_parser.add_argument('-o',
                           action='store_true',
                           dest='outlier_detection',
                           help='Evaluate using outlier detection.')

    my_parser.add_argument('-c',
                           metavar='PATH',
                           action='store',
                           type=str,
                           dest='cross_match_emb',
                           help='Compare first word embedding with the embedding from the given path.')

    my_parser.add_argument('-v',
                           '--verbose',
                           action='store_true',
                           help='')

    # Execute the parse_args() method
    args = my_parser.parse_args()
    print(vars(args))

    path_embedding = args.path
    embedding = getWordEmbedding(path_embedding)
    word_embedding = Embedding(embedding)

    similarity_datasets = args.similarity_datasets
    analogy_datasets = args.analogy_datasets
    outlier = args.outlier_detection
    cross_match_emb = args.cross_match_emb

    if similarity_datasets != None:
        print("Word similarity test")
        similarity_test = SimilarityTestClass(datasets=similarity_datasets)
        similarity_test.evaluateWordVector("test_script", word_embedding)

    if analogy_datasets != None:
        print("Word analogy test")
        analogy_test = AnalogyTestClass(datasets=analogy_datasets, metrics=["3CosMul"])
        analogy_test.evaluateWordVector("test_script", word_embedding)

    if outlier:
        print("Outlier detection test")
        outlier_detection_test = OutlierDetectionTestClass()
        outlier_detection_test.evaluateWordVector("test_script", word_embedding)

    if cross_match_emb != None:
        print("Cross-match test")

        word_embedding2 = getWordEmbedding(cross_match_emb)
        cross_match_test = CrossMatchTestClass()
        cross_match_test.crossMatchTest(word_embedding, "test_script1", word_embedding2, "test_script2")


if __name__ == '__main__':
    main()