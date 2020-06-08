from gensim.models.keyedvectors import KeyedVectors

import shutil

import os
import io
import numpy as np

from scipy.stats import spearmanr

import Constant

# Dataset y resultados
_DATASET = Constant.DATA_FOLDER / "SimilarityDataset"
_RESULT = Constant.RESULTS_FOLDER / "Similarity"
_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "Similarity"

# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file

    return KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)

class SimilarityTestClass:
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _lower = True
    _use_intersect_dataset = False
    _oov_word = {}

    def __init__(self, cantidad=None, lower=True, use_intersect_dataset=False):
        print("Test de Similaridad")

        self._embeddings_size = cantidad
        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset

    ###########################################################################################
    # METRICA
    ###########################################################################################

    """
    Obtencion de correlacion spearman rho a partir de pares de palabras y puntaje de similaridad

    :param embedding: lista de vectores de palabras
    :param word_pairs: lista de pares de palabras con su puntaje de similaridad

    :return: lista con correlacion sperman-rho, cantidad de pares evaluados, cantidad de pares no evaluados
             y cantidad de palabras no encontradas en el vocabulario
    """
    def get_spearman_rho(self, embedding, word_pairs):
        not_found_pairs = 0
        not_found_words = 0
        not_found_list = []

        pred = []
        gold = []
        for word1, word2, similarity in word_pairs:
            w1 = word1 in embedding
            w2 = word2 in embedding

            u = np.array([])
            if not w1:
                not_found_pairs += 1
                not_found_words += 1
                not_found_list.append(word1)
                if word1 not in self._oov_word.keys():
                    self._oov_word[word1] = np.random.rand(embedding.vector_size)

                u = self._oov_word[word1]

            else:
                u = embedding[word1]

            v = np.array([])
            if not w2:
                if w1:
                    not_found_pairs += 1
                not_found_words += 1
                not_found_list.append(word2)
                if word2 not in self._oov_word.keys():
                    self._oov_word[word2] = np.random.rand(embedding.vector_size)

                v = self._oov_word[word2]

            else:
                v = embedding[word2]

            # Calculo de similaridad coseno
            score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
            gold.append(similarity)
            pred.append(score)

        print("    Not found words:" + str(not_found_words))
        for word in not_found_list:
            print("    " + word)

        return spearmanr(gold, pred).correlation, len(gold), not_found_pairs, not_found_words


    ###########################################################################################
    # MANEJO DE ARCHIVOS Y DATASET
    ###########################################################################################

    def intersectDataset(self, word_vector):
        # TODO: revisar si hay archivos en la carpeta de interseccion
        # TODO: realizar interseccion de dataset con dataset de interseccion o dataset original
        next_dataset_path = Constant.DATA_FOLDER / "_intersect_SimilarityDataset"
        if not next_dataset_path.exists():
            os.makedirs(next_dataset_path)

        if len(os.listdir(next_dataset_path)) == 0:
            for file_name in os.listdir(_DATASET):
                origin_file = _DATASET / file_name
                shutil.copy(origin_file, next_dataset_path)

        for file_name in os.listdir(next_dataset_path):
            file_path = next_dataset_path / file_name
            lines = []
            with io.open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tupla = line.split()
                    if tupla[0] not in word_vector or tupla[1] not in word_vector:
                        continue

                    lines.append(line)

            with io.open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line)


    """
    Obtencion de archivos de test desde carpeta de datasets
    
    :return: lista de nombres de dataset
    """
    def getTestFiles(self):
        if not _DATASET.exists():
            raise Exception("No se logro encontrar carpeta con test")

        return os.listdir(_DATASET)


    """
    Obtencion de pares de palabras en algun archivo
    
    :param file: nombre de archivo test
    :param lower: determina si se cambian las mayusculas por minusculas
    
    :return: lista con pares de palabras y su puntaje de similaridad
    """
    def getWordPairs(self, file, lower):
        print(">>> Abriendo archivo " + file)

        word_pairs = []
        total = 0
        with io.open(_DATASET / file, 'r', encoding='utf-8') as f:
            for line in f:
                total = total + 1

                line = line.strip()
                line = line.lower() if lower else line
                line = line.split()

                # Ignorar frases y solo considerar palabras
                if len(line) != 3:
                    assert len(line) > 3
                    continue

                pair = (line[0], line[1], float(line[2]))
                word_pairs.append(pair)

        return word_pairs


    ###########################################################################################
    # GUARDAR RESULTADOS
    ###########################################################################################


    """
    Guarda resultados del test de similaridad
    
    :param embedding_name: nombre del embedding
    :param score: lista de resultados
    """
    def saveResults(self, embedding_name, score):
        if not _RESULT.exists():
            os.makedirs(_RESULT)

        result_path = _RESULT / (embedding_name + ".txt")
        print(">>> Guardando resultados en:\n    " + str(result_path))

        with io.open(result_path, 'w', encoding='utf-8') as f:
            for tuple in score:
                for key in tuple.keys():
                    f.write(key + ": " + str(tuple[key]) + "\n")


    ###########################################################################################
    # EVALUACION POR SIMILARITY
    ###########################################################################################


    """
    Realizacion de test de similaridad
    
    :return: coeficiente de correlacion spearman rho, cantidad de palabras no encontradas y cantidad de pares no evaluados
    """
    def similarityTest(self):
        results = {}
        if self._use_intersect_dataset:
            print("Obteniendo interseccion de datasets")
            for embedding_name in self._embeddings_name_list:
                word_vector = get_wordvector(embedding_name, self._embeddings_size)
                self.intersectDataset(word_vector)

                _DATASET = Constant.DATA_FOLDER / "_intersect_SimilarityDataset"

        # Realizacion de test por cada embedding
        for embedding_name in self._embeddings_name_list:
            word_vector_name = embedding_name.split('.')[0]
            print(">>> Cargando vectores" + word_vector_name + "...", end='')
            word_vector = get_wordvector(embedding_name, self._embeddings_size)
            print("listo.\n")


            # Obtencion de nombre de archivos de test
            test_file_list = self.getTestFiles()
            all_word_pairs = []
            scores = []


            # Test en archivos individuales
            print(">>> Test individuales")
            for test_file in test_file_list:
                print("    Archivo test: " + test_file)
                word_pairs = self.getWordPairs(test_file, self._lower)
                all_word_pairs = all_word_pairs + word_pairs

                coeff, found, not_found_pairs, not_found_words = self.get_spearman_rho(word_vector, word_pairs)
                scores.append({
                    test_file: coeff,
                    "not_found_pairs": not_found_pairs,
                    "not_found_words": not_found_words,
                })

                print("    > Cantidad de pares no procesados: " + str(not_found_pairs) + "\n\n")

            # Test en conjunto
            print(">>> Empezando test en conjunto")

            coeff, found, not_found_pairs, not_found_words = self.get_spearman_rho(word_vector, all_word_pairs)
            scores.append({
                "all_data": coeff,
                "not_found_pairs": not_found_pairs,
                "not_found_words": not_found_words
            })

            print("    > Cantidad de pares no procesados: " + str(not_found_pairs) + "\n\n")

            # Guardando resultados
            self.saveResults(word_vector_name, scores)
            results[word_vector_name] = scores

            print(">>> Resultados")
            for tuple in scores:
                print(tuple)

            print("\n")

        return results


