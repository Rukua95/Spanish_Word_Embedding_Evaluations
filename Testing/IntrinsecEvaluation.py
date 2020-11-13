#!/usr/bin/env python

import os
import io
import sys
import argparse

from pathlib import Path
from gensim.models.keyedvectors import KeyedVectors

from SimilarityTest import SimilarityTestClass
from AnalogyTest import AnalogyTestClass
from OutlierDetectionTest import OutlierDetectionTestClass
from CrossMatchTest import CrossMatchTestClass

# Clase utilizada para representar embeddings
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

# Metodo para cargar embeddings
def getWordEmbedding(embedding_path):
    print("> Loading word embedding...")
    try:
        word_embedding = KeyedVectors.load_word2vec_format(embedding_path, limit=None)
    except:
        raise Exception("Is not posible to load the word embedding: " + str(embedding_path))
    print("  Done loading word embedding")

    return word_embedding

# Evalucion comparativa de multiples word embeddings
def multipleEmbeddingEval(args):
    # Verificar path a directorio de embeddings.
    try:
        word_embeddings_directory = Path(args.embedding_directory)
        embeddings_files_name = os.listdir(word_embeddings_directory)
    except:
        raise Exception("Problems at reading path directory with embeddings: " + str(word_embeddings_directory))

    # Obtencion de argumentos
    similarity_datasets = args.similarity_datasets
    analogy_datasets = args.analogy_datasets
    outlier = args.outlier_detection
    cross_match_emb = args.cross_match_emb

    verbose = args.verbose

    # Verificar existencia de path de segundo word embedding para cross-match
    try:
        if cross_match_emb != None:
            cross_match_emb = Path(args.cross_match_emb)
    except:
        raise Exception("Cross-match second embedding path is invalid: " + str(cross_match_emb))

    # Creacion de test de similaridad
    if similarity_datasets != None:
        similarity_test = SimilarityTestClass(use_intersect_dataset=True, datasets=similarity_datasets)
        similarity_test.resetIntersectDataset()

    # Creacion de test de analogias
    if analogy_datasets != None:
        analogy_test = AnalogyTestClass(use_intersect_dataset=True, datasets=analogy_datasets, metrics=["3CosMul"])
        analogy_test.resetIntersectDataset()

    # Creacion de test de outlier detection
    if outlier:
        outlier_detection_test = OutlierDetectionTestClass(use_intersect_dataset=True)
        outlier_detection_test.resetIntersectDataset()

    # Interseccion de vocabularios
    for embedding_file in embeddings_files_name:
        print("> Intersecting dataset with", (word_embeddings_directory / embedding_file).stem, "vocabulary...")
        embedding = getWordEmbedding(word_embeddings_directory / embedding_file)
        word_embedding = Embedding(embedding)

        if similarity_datasets != None:
            similarity_test.intersectDataset(word_embedding)

        if analogy_datasets != None:
            analogy_test.intersectDataset(word_embedding)

        if outlier:
            outlier_detection_test.intersectDataset(word_embedding)

        print("  Done intersecting")
        del word_embedding

    # Evaluacion de embeddings
    for embedding_file in embeddings_files_name:
        embedding_path = word_embeddings_directory / embedding_file

        embedding = getWordEmbedding(embedding_path)
        word_embedding = Embedding(embedding)
        word_embedding_name = embedding_path.stem

        # Evaluacion por similaridad
        if similarity_datasets != None:
            print("> Word similarity test")
            results = similarity_test.evaluateWordVector(word_embedding_name, word_embedding)
            print("  Evaluation complete\n")

            if verbose:
                print("> Results & Info")
                for r in results:
                    print(r)

                print("\n")

        # Evaluacion por analogias
        if analogy_datasets != None:
            print("> Word analogy test")
            results = analogy_test.evaluateWordVector(word_embedding_name, word_embedding)
            print("  Evaluation complete")

            if verbose:
                print("> Results & Info")
                for r in results:
                    print(r)

                print("\n")

        # Evaluacion por outlier detection
        if outlier:
            print("> Outlier detection test")
            results = outlier_detection_test.evaluateWordVector(word_embedding_name, word_embedding)
            print("  Evaluation complete")

            if verbose:
                print("> Results & Info")
                for r in results:
                    print(r)

                print("\n")

        # Evaluacion por cross-match
        if cross_match_emb != None:
            print("> Cross-match test")
            word_embedding2 = getWordEmbedding(cross_match_emb)
            word_embedding_name2 = cross_match_emb.stem

            cross_match_test = CrossMatchTestClass()
            results = cross_match_test.crossMatchTest(word_embedding, word_embedding_name, word_embedding2,
                                                      word_embedding_name2)
            print("  Evaluation complete")

            if verbose:
                print("> Results & Info")
                for r in results:
                    print(r)

                print("\n")

# Evaluacion individual de word embeddings
def singleEmbeddingEval(args):
    # Verificar path a word embedding
    try:
        word_embedding_path = Path(args.embedding_file)
        word_embedding_name = word_embedding_path.stem
    except:
        raise Exception("Problems at reading path directory with embeddings: " + str(word_embedding_path))

    embedding = getWordEmbedding(word_embedding_path)
    word_embedding = Embedding(embedding)

    # Argumentos
    similarity_datasets = args.similarity_datasets
    analogy_datasets = args.analogy_datasets
    outlier = args.outlier_detection
    cross_match_emb = args.cross_match_emb

    verbose = args.verbose

    # Verificar path a word embedding utilizado en cross-match
    try:
        if cross_match_emb != None:
            cross_match_emb = Path(args.cross_match_emb)
    except:
        raise Exception("Cross-match second embedding path is invalid")

    # Evalucion por similaridad
    if similarity_datasets != None:
        print("> Word similarity test")
        similarity_test = SimilarityTestClass(datasets=similarity_datasets)
        results = similarity_test.evaluateWordVector(word_embedding_name, word_embedding)
        print("  Evaluation complete\n")

        if verbose:
            print("> Results & Info")
            for r in results:
                print(r)

            print("\n")

    # Evaluacion por analogias
    if analogy_datasets != None:
        print("> Word analogy test")
        analogy_test = AnalogyTestClass(datasets=analogy_datasets, metrics=["3CosMul"])
        results = analogy_test.evaluateWordVector(word_embedding_name, word_embedding)
        print("  Evaluation complete")

        if verbose:
            print("> Results & Info")
            for r in results:
                print(r)

            print("\n")

    # Evaluacion por outlier detection
    if outlier:
        print("> Outlier detection test")
        outlier_detection_test = OutlierDetectionTestClass()
        results = outlier_detection_test.evaluateWordVector(word_embedding_name, word_embedding)
        print("  Evaluation complete")

        if verbose:
            print("> Results & Info")
            for r in results:
                print(r)

            print("\n")

    # Evaluacion por cross-match
    if cross_match_emb != None:
        print("> Cross-match test")
        word_embedding2 = getWordEmbedding(cross_match_emb)
        word_embedding2 = Embedding(word_embedding2)
        word_embedding_name2 = cross_match_emb.stem

        cross_match_test = CrossMatchTestClass()
        results = cross_match_test.crossMatchTest(word_embedding, word_embedding_name, word_embedding2,
                                                  word_embedding_name2)
        print("  Evaluation complete")

        if verbose:
            print("> Results & Info")
            for r in results:
                print(r)

            print("\n")

def main():

    # Parser
    my_parser = argparse.ArgumentParser(description='Word embedding intrinsec evaluation program. Word embeddings are loaded '
                                                    'using load_word2vec_format from KeyedVectors in gensim library. '
                                                    'Results are saved in folder "Resultados", under the same name as the '
                                                    'file containing the embedding. This program support evaluation by '
                                                    'word similarity, word analogy, outlier detection and cross-match.')

    my_group = my_parser.add_mutually_exclusive_group(required=True)

    # Argumentos
    my_group.add_argument('-d',
                          metavar='PATH',
                          type=str,
                          action='store',
                          dest='embedding_directory',
                          help='Path to directory containing word embeddings to evaluate.')

    my_group.add_argument('-f',
                          metavar='PATH',
                          type=str,
                          action='store',
                          dest='embedding_file',
                          help='Path to word embedding to evaluate.')

    my_parser.add_argument('-a',
                           action='store',
                           metavar='DATASET',
                           type=str,
                           nargs='*',
                           dest='analogy_datasets',
                           help='Evaluate using word analogy. If no dataset is given, the evaluation use all '
                                'dataset avalible. By default, only 3CosMul is used during evaluation.')

    my_parser.add_argument('-s',
                           action='store',
                           metavar='DATASET',
                           type=str,
                           nargs='*',
                           dest='similarity_datasets',
                           help='Evaluate using semantic similarity. If no dataset is given, the evaluation '
                                'use all dataset avalible. Pearson r, Spearman rho and Kendall tau are used during '
                                'evaluation.')

    my_parser.add_argument('-o',
                           action='store_true',
                           dest='outlier_detection',
                           help='Evaluate using outlier detection.')

    my_parser.add_argument('-c',
                           metavar='PATH',
                           action='store',
                           type=str,
                           dest='cross_match_emb',
                           help='Compare first word embedding with the embedding from the given path. In case '
                                'first path given is a folder, cross-match evaluation is done with all embeddings '
                                'from folder.')

    my_parser.add_argument('-v',
                           '--verbose',
                           dest='verbose',
                           action='store_true',
                           help='Directly show results from evaluations and related info.')

    args = my_parser.parse_args()

    word_embedding_path = args.embedding_file
    word_embeddings_directory = args.embedding_directory

    if word_embedding_path != None:
        singleEmbeddingEval(args)
    elif word_embeddings_directory != None:
        multipleEmbeddingEval(args)

if __name__ == '__main__':
    main()