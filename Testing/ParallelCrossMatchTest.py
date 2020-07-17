import CrossMatchTest

from gensim.models.keyedvectors import KeyedVectors

import Constant

import random
import math
import networkx as nx

import shutil
import os
import io
import numpy as np

import Constant

from multiprocessing import Pool
from functools import partial


# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file
    print(">>> Cargando vectores " + file + " ...", end='')
    word_vector = KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)
    print("listo.\n")

    return word_vector

class CrossMatchTestClass:
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _main_sample = 100000
    _sub_sample_size = 200
    _repetitions = 500

    # Dataset y resultados
    _RESULT = Constant.RESULTS_FOLDER / "CrossMatch"

    def __init__(self, cantidad=None, main_sample=100000, sub_sample_size=200, repetitions=500):
        print("Test de CrossMatch (paralelo)")

        self._crossMatch = CrossMatchTest.CrossMatchTestClass(cantidad, sub_sample_size, repetitions)

        self._embeddings_size = cantidad
        self._main_sample = main_sample
        self._sub_sample_size = sub_sample_size
        self._repetitions = repetitions


    def saveResult(self, embedding1_name, embedding2_name, results):
        if not self._RESULT.exists():
            os.makedirs(self._RESULT)

        result_file = self._RESULT / (embedding1_name + "_" + embedding2_name + ".txt")
        with io.open(result_file, 'w', encoding='utf-8') as f:
            f.write(str(results) + "\n")


    def crossMatchTest(self):
        results = {}

        for i in range(len(self._embeddings_name_list)):
            embedding_name1 = self._embeddings_name_list[i]
            word_vector_name1 = embedding_name1.split('.')[0]
            word_vector1 = get_wordvector(embedding_name1, self._embeddings_size)
            word_list1 = self._crossMatch.getWordList(word_vector1)

            for j in range(len(self._embeddings_name_list)):
                if j <= i:
                    continue

                embedding_name2 = self._embeddings_name_list[j]
                word_vector_name2 = embedding_name2.split('.')[0]
                word_vector2 = get_wordvector(embedding_name2, self._embeddings_size)
                word_list2 = self._crossMatch.getWordList(word_vector2)

                sum = 0

                #TODO: paralelizar esta accion
                print("Testing en paralelo de" + word_vector_name1 + " y " + word_vector_name2)
                input_G = []
                output_M = []
                print(">>> Getting graphs")
                for h in range(self._repetitions):
                    #print("matrix", h)
                    matrix = self._crossMatch.getDistanceMatrix(word_vector1, word_vector2, word_list1, word_list2)
                    #for l in matrix:
                    #    print(l)

                    G = self._crossMatch.getGraph(matrix)
                    input_G.append(G)

                print(">>> Empezando procesos en paralelo")
                with Pool(4) as p:
                    func = partial(nx.max_weight_matching, maxcardinality=True)
                    output_M = p.map(func, input_G)
                print(">>> Terminando procesos en paralelo")

                for h in range(len(output_M)):
                    M = output_M[h]

                    p_value, c1 = self._crossMatch.getScore(M)
                    if (h+1) % 10 == 0:
                        print("p-value " + str(h) + ": " + str(p_value), "c1 =", c1)

                    sum += p_value

                pair_result = sum / self._repetitions
                results[word_vector_name1 + "__" + word_vector_name2] = pair_result
                self.saveResult(word_vector_name1, word_vector_name2, pair_result)

                print("Result: " + str(pair_result), end='\n\n')

        return results