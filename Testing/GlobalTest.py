import SimilarityTest, AnalogyTest, OutlierDetectionTest, CrossMatchTest, ConstitucionUtil

import EvalConstitucion

import os
import io
import json

import numpy as np

from scipy.stats import spearmanr
from gensim.models.keyedvectors import KeyedVectors

import Constant

# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file

    return KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)

###################################################################################

class GlobalTest:
    _embeddings_name = os.listdir(EMBEDDING_FOLDER)
    _extra_embeddings = []
    _intrinsec_test_settings = {
        "lower": True,
        "similarity": {
        },
        "analogy": {
            "all_score": False,
            "all_combination": False,
        },
        "outlierdetection": {
            "exist_oov": True
        },
        "crossmatch": {
            "enable": True,
            "repetition": 500,
            "sample_size": 100000,
            "sub_sample_size": 200,
        },
    }
    _results = []

    def __init__(self, extra_embeddings=[]):
        self._extra_embeddings = extra_embeddings

        print("Embeddings defaults a evaluar:")
        for name in self._embeddings_name:
            print("> " + name)

        print("Embeddings extras a evaluar:")
        for name in self._extra_embeddings:
            print("> " + name)


    # Evaluaciones intrinsecas
    def similarityTest(self, word_vector, word_vector_name):
        result = SimilarityTest.similarityTest(word_vector, word_vector_name, self._lower)
        return result


    def analogyTest(self, word_vector, word_vector_name):
        all_combination = self._intrinsec_test_settings["analogy"]["all_combination"]
        all_score = self._intrinsec_test_settings["analogy"]["all_score"]

        result = AnalogyTest.analogyTest(word_vector, word_vector_name, all_score, all_combination, self._lower)
        return result


    def outlierDetectionTest(self, word_vector, word_vector_name):
        exist_oov = self._intrinsec_test_settings["outlierdetection"]["exist_oov"]

        result = OutlierDetectionTest.outlierDetectionTest(word_vector, word_vector_name, exist_oov, self._lower)
        return result


    def crossMatchTest(self, word_vector1, word_vector1_name, word_vector2, word_vector2_name):
        # TODO: obtener valores desde setting
        sample_size = self._intrinsec_test_settings["crossmatch"]["sample_size"]
        sub_sample_size = self._intrinsec_test_settings["crossmatch"]["sub_sample_size"]
        repetition = self._intrinsec_test_settings["crossmatch"]["repetition"]
        F_constant = CrossMatchTest.getFConstant(sub_sample_size)

        result = CrossMatchTest.crossMatchTest(word_vector1, word_vector1_name, word_vector2, word_vector2_name,
                                               sample_size, sub_sample_size, repetition, F_constant)
        return result


    # Evaluaciones extrinseca
    def constitutionClassificationTask(self, word_vector, word_vector_name):
        mean_vector_results = EvalConstitucion.MeanVectorEvaluation(word_vector, word_vector_name)
        #RRN_results = EvalConstitucion.RNNEvaluation(word_vector, word_vector_name)

        return mean_vector_results


    # Evaluacion global
    def globalTest(self, number_of_vectors=None):
        print("Iniciando test intrinsecos")
        all_embeddings = self._embeddings_name + self._extra_embeddings

        for i in range(len(all_embeddings)):
            name = self._embeddings_name[i]
            print(">>> Testing " + name + "\n")

            print(">>> Cargando vectores...", end='')
            word_vector_path = EMBEDDING_FOLDER / name
            word_vector_name = name
            if i >= len(self._embeddings_name):
                word_vector_path = name
                word_vector_name = ("extra_embedding_" + str(len(self._embeddings_name) - i))

            word_vector = KeyedVectors.load_word2vec_format(word_vector_path, limit=number_of_vectors)

            print("listo")

            print("\nIniciando test de similaridad")
            similarityResult = self.similarityTest(word_vector, word_vector_name)

            print("\nIniciando test de analogias")
            analogyResult = self.analogyTest(word_vector, word_vector_name)

            print("\nIniciando test de outlier detection")
            outlierDetectionResult = self.outlierDetectionTest(word_vector, word_vector_name)

            if self._intrinsec_test_settings["crossmatch"]["enable"]:
                print("\nIniciando test de cross-match")

                crossMatchResult = []
                for j in range(len(all_embeddings)):
                    if j <= i:
                        continue

                    name2 = self._embeddings_name[j]
                    print(">>> Comparando con " + name2 + "\n")

                    print(">>> Cargando vectores...", end='')
                    word_vector_path2 = EMBEDDING_FOLDER / name2
                    word_vector_name2 = name2
                    if i >= len(self._embeddings_name):
                        word_vector_path2 = name2
                        word_vector_name2 = ("extra_embedding_" + str(len(self._embeddings_name) - i))

                    word_vector2 = KeyedVectors.load_word2vec_format(word_vector_path2, limit=number_of_vectors)

                    print("listo")

                    crossMatchResult.append(self.crossMatchTest(word_vector, word_vector_name, word_vector2, word_vector_name2))

            print("Iniciando test extrinseco")
            constitucionResult = self.constitutionClassificationTask(word_vector, word_vector_name)


    def changeSettings(self, test_name, test_setting, new_value):
        if test_name in self._intrinsec_test_settings.keys():
            if test_setting in self._intrinsec_test_settings[test_name].keys():
                self._intrinsec_test_settings[test_name][test_setting] = new_value