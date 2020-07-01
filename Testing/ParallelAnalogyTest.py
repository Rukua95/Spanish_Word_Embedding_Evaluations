from gensim.models.keyedvectors import KeyedVectors
import AnalogyTest

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

class AnalogyTestClass:
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _lower = True
    _use_intersect_dataset = False
    _oov_word = []

    _all_score = False
    _all_combination = False

    # Archivos que no se pueden usar para relizar test en ambas direcciones de la relacion.
    RESTRICTED_FILES = [
        "_español_E01 [pais - capital].txt",
        "_español_E02 [pais - idioma].txt",
        "_español_E04 [nombre - nacionalidad].txt",
        "_español_E05 [nombre - ocupacion].txt",
        "_español_E11 [ciudad_Chile - provincia_Chile].txt",
        "_español_E12 [ciudad_EEUU - estado_EEUU].txt",
        "_español_L07 [sinonimos - intensidad].txt",
        "_español_L08 [sinonimos - exacto].txt",
        "_español_L09 [antonimos - grado].txt",
        "_español_L10 [antonimos - binario].txt",
    ]

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "AnalogyDataset"
    _RESULT = Constant.RESULTS_FOLDER / "Analogy"
    # TODO: crear una carpeta de resultados temporales en la carpeta de resultados
    _TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "Analogy"

    def __init__(self, cantidad=None, lower=True, use_intersect_dataset=False, all_score=False, all_combination=False):
        print("Test de Analogias")

        self._analogy = AnalogyTest.AnalogyTestClass()

        self._embeddings_size = cantidad
        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset

        self._all_score = all_score
        self._all_combination = all_combination

    def analogyTest(self):
        results = {}

        # Interseccion de datasets
        if self._use_intersect_dataset:
            print("Obteniendo interseccion de datasets")
            for embedding_name in self._embeddings_name_list:
                word_vector = get_wordvector(embedding_name, self._embeddings_size)
                state = self._analogy.intersectDataset(word_vector)

                if not state:
                    raise Exception("Interseccion vacia de embeddings, no se puede continuar con la evaluacion")

            self._DATASET = Constant.DATA_FOLDER / "_intersection_AnalogyDataset"
            self._RESULT = Constant.RESULTS_FOLDER / "_intersection_Analogy"

        else:
            self._DATASET = Constant.DATA_FOLDER / "AnalogyDataset"
            self._RESULT = Constant.RESULTS_FOLDER / "Analogy"

        # Realizacion de test por cada embedding
        for embedding_name in self._embeddings_name_list:
            embedding_results = []
            word_vector_name = embedding_name.split('.')[0]
            word_vector = get_wordvector(embedding_name, self._embeddings_size)

            # Obtencion de archivos que faltan por testear
            test_file_list = self._analogy.getUntestedFiles(word_vector_name)

            # Revisamos todos los archivos para realizar test
            output_file_results = []
            with Pool(4) as p:
                func = partial(self._analogy.evaluateFile, word_vector=word_vector)
                output_file_results = p.map(func, test_file_list)

            print("Fin de test para " + embedding_name)

            assert len(test_file_list) == len(output_file_results)
            for i in range(len(test_file_list)):
                total_test_result = {}
                file = test_file_list[i]
                result = output_file_results[i]
                pair_list = self._analogy.getAnalogyPairs(file)

                print("Guardando resultados de " + file)

                count_multiply = 1
                count_relations = len(pair_list) * (len(pair_list) - 1) / 2

                # En caso de evaluar todas las combinaciones, diferenciamos los test que tienen relaciones no biyectivas
                if self._all_combination:
                    if not file.name in self.RESTRICTED_FILES:
                        count_multiply = 4
                    else:
                        count_multiply = 2

                similarity_total = result[0]
                che_metric = result[1]
                oov_tuples = result[2]
                oov_elements = result[3]

                # Calculamos los resultados totales del test
                # Similaridad
                total_test_result["3CosAdd"] = (
                            similarity_total[0] / (count_relations * count_multiply)) if count_relations > 0 else "Nan"
                total_test_result["3CosMul"] = (
                            similarity_total[1] / (count_relations * count_multiply)) if count_relations > 0 else "Nan"
                total_test_result["PairDir"] = (
                            similarity_total[2] / (count_relations * count_multiply)) if count_relations > 0 else "Nan"

                # Puntajes definido por Che
                total_test_result["cos"] = (che_metric[0] / count_relations) if count_relations > 0 else "Nan"
                total_test_result["euc"] = (che_metric[1] / count_relations) if count_relations > 0 else "Nan"
                total_test_result["ncos"] = (che_metric[2] / count_relations) if count_relations > 0 else "Nan"
                total_test_result["neuc"] = (che_metric[3] / count_relations) if count_relations > 0 else "Nan"

                # Estadisticas de palabras/elementos oov
                total_test_result["%oov_tuplas"] = (oov_tuples / count_relations) if count_relations > 0 else "Nan"
                total_test_result["%oov_elements"] = (
                            oov_elements / (4 * count_relations)) if count_relations > 0 else "Nan"

                # Guardamos los resultados de forma temporal
                embedding_results[file.name] = total_test_result


        return results