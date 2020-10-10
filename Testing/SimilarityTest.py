from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import spearmanr, kendalltau, pearsonr

import numpy as np

import shutil
import os
import io
import Constant


# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file
    print(">>> Cargando vectores " + file + " ...", end='')
    word_vector = KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)
    print("listo.\n")

    return word_vector


class SimilarityTestClass:
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "SimilarityDataset"

    _INTERSECT_DATASET = Constant.DATA_FOLDER / "_intersection_SimilarityDataset"
    _ORIGINAL_DATASET = Constant.DATA_FOLDER / "SimilarityDataset"

    _RESULT = Constant.RESULTS_FOLDER / "Similarity"

    def __init__(self, cantidad=None, lower=True, use_intersect_dataset=False):
        print("Test de Similaridad")

        self._embeddings_size = cantidad
        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset

        if self._use_intersect_dataset:
            self._DATASET = self._INTERSECT_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "_intersection_Similarity"

        else:
            self._DATASET = self._ORIGINAL_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "Similarity"

    ###########################################################################################
    # EVALUACION
    ###########################################################################################

    """
    Obtencion de correlacion Pearson r, Spearman rho y Kendall tau a partir de pares de palabras y puntaje de similaridad

    :param embedding: lista de vectores de palabras
    :param word_pairs: lista de pares de palabras con su puntaje de similaridad

    :return: lista con correlacion sperman-rho, cantidad de pares evaluados, cantidad de pares no evaluados
             y cantidad de palabras no encontradas en el vocabulario
    """
    def evaluate(self, embedding, word_pairs):
        not_found_pairs = 0
        not_found_words = 0
        repeated_pairs = 0

        not_found_list = []

        pred = []
        gold = []
        reg = []
        for word1, word2, similarity in word_pairs:
            w1 = word1 in embedding
            w2 = word2 in embedding

            aux = [word1, word2]
            aux.sort()
            if aux in reg:
                repeated_pairs += 1
                print("Par repetido: ", word1, " ", word2)
                continue
            else:
                reg.append(aux)

            u = np.array([])
            v = np.array([])
            if not w1 or not w2:
                not_found_pairs += 1
                not_found_words += (w1 + w2)

                if not w1:
                    not_found_list.append(word1)

                if not w2:
                    not_found_list.append(word2)

                continue

            else:
                u = embedding[word1]
                v = embedding[word2]

            # Calculo de similaridad coseno
            score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
            gold.append(similarity)
            pred.append(score)

        print("    Pares repetidos:", repeated_pairs)
        print("    Not found words: " + str(not_found_words))
        print("    ", end='')
        print(not_found_list)

        # Lista de resultados
        p_r = ["pearson", pearsonr(gold, pred)[0]]
        s_rho = ["spearman", spearmanr(gold, pred)[0]]
        k_tau = ["kendall", kendalltau(gold, pred)[0]]

        res = [p_r, s_rho, k_tau]

        return res, not_found_pairs, not_found_words


    ###########################################################################################
    # MANEJO DE ARCHIVOS Y DATASET
    ###########################################################################################

    """
    Metodo para eliminar dataset obtenido a travez de intersectar vocabulario de embeddings
    """
    def resetIntersectDataset(self):
        print("Eliminando archivos en carpeta de interseccion de dataset")
        intersect_dataset_path = self._INTERSECT_DATASET
        if intersect_dataset_path.exists():
            shutil.rmtree(intersect_dataset_path)


    """
    Metodo que crea dataset con la interseccion de vocabulario de los embeddings guardados
    """
    def createIntersectDataset(self):
        print("Obteniendo interseccion de datasets")

        for embedding_name in self._embeddings_name_list:
            word_vector = get_wordvector(embedding_name, self._embeddings_size)
            state = self.intersectDataset(word_vector)

            if not state:
                raise Exception("Interseccion vacia de embeddings, no se puede continuar con la evaluacion")

        print("Nuevo dataset en:\n ", str(self._INTERSECT_DATASET))


    """
    Metodo que elimina palabras de los dataset, que esten fuera del vocabulario del word embeddings dado
    
    :param word_vector: word embedding para comparar vocabulario
    """
    def intersectDataset(self, word_vector):
        print("Intersectando datasets...")
        next_dataset_path = self._INTERSECT_DATASET
        deleted_files = 0

        # Verificar que existe carpeta para guardar nuevo dataset
        if not next_dataset_path.exists():
            os.makedirs(next_dataset_path)

        # Verificar si hay datasets ya intersectados
        print(" > Revisando si existe interseccion previa")
        for file_name in os.listdir(self._ORIGINAL_DATASET):
            if file_name in os.listdir(next_dataset_path):
                print("   > ", file_name, " ya ha sido intersectado anteriormente")
            else:
                origin_file = self._ORIGINAL_DATASET / file_name
                shutil.copy(origin_file, next_dataset_path)
                print("   > ", file_name, " no ha sido intersectado anteriormente, copiando")

        # Revisar cada archivo dentro de la carpeta de dataset
        print(" > Revision de archivos en dataset")
        to_delete_files = []
        for file_name in os.listdir(next_dataset_path):
            print(" > Revisando " + file_name)
            file_path = next_dataset_path / file_name
            deleted_element = 0
            lines = []

            # Revisar el dataset intersectado que llevamos hasta el momento
            with io.open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tupla = line.lower().split()

                    # Revisar cantidad de palabras en tupla
                    if len(tupla) != 3:
                        deleted_element += 1
                        continue

                    # Revisar que todas las palabras estan en el vocabulario
                    if tupla[0] not in word_vector or tupla[1] not in word_vector or len(tupla) != 3:
                        deleted_element += 1
                        continue

                    lines.append(line)

            # Eliminamos archivo que no aporta al analisis
            if len(lines) == 0:
                deleted_files += 1
                to_delete_files.append(file_path)
                print("  > Archivo esta vacio, se procede a eliminar")
                continue

            # Escribir documento
            with io.open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line)

            print(" > lineas eliminadas: " + str(deleted_element) + " de " + str(deleted_element + len(lines)))

        print(" > archivos a eliminar: " + str(deleted_files) + "\n")
        for file in to_delete_files:
            os.remove(file)

        return True if len(os.listdir(next_dataset_path)) > 0 else False


    """
    Obtencion de archivos de test desde carpeta de datasets
    
    :return: lista de nombres de dataset
    """
    def getTestFiles(self):
        print(">>> Obteniendo nombre de archivos de test desde:\n     " + str(self._DATASET))
        if not self._DATASET.exists():
            raise Exception("No se logro encontrar carpeta con test")

        return os.listdir(self._DATASET)


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
        with io.open(self._DATASET / file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.lower() if lower else line
                line = line.split()

                # Ignorar frases y solo considerar palabras
                if len(line) != 3:
                    assert len(line) > 3
                    continue

                total = total + 1

                pair = (line[0], line[1], float(line[2]))
                word_pairs.append(pair)

            print("    lineas encontradas en archivo: " + str(total))

        return word_pairs


    """
    Guarda resultados del test de similaridad
    
    :param embedding_name: nombre del embedding
    :param score: lista de resultados
    """
    def saveResults(self, embedding_name, score):
        save_path = self._RESULT

        if not save_path.exists():
            os.makedirs(save_path)

        result_path = save_path / (embedding_name + ".txt")
        print(">>> Guardando resultados en:\n     " + str(result_path))

        with io.open(result_path, 'w', encoding='utf-8') as f:
            for res in score:
                for data in res:
                    for el in data:
                        f.write(str(el) + " ")

                    f.write("\n")


    ###########################################################################################
    # EVALUACION POR SIMILARITY
    ###########################################################################################

    """
    Evalua un word embedding especifico y guarda el resultado en carpeta de resultados

    :param word_vector_name: nombre de word embedding
    :param word_vector: word embedding a evaluar
    """

    def evaluateWordVector(self, word_vector_name, word_vector):
        # Obtencion de nombre de archivos de test
        test_file_list = self.getTestFiles()
        scores = []

        # Test en archivos individuales
        print(">>> Test individuales")
        for test_file in test_file_list:
            word_pairs = self.getWordPairs(test_file, self._lower)

            # Evaluamos embeddings con dataset especifico
            coeffs, not_found_pairs, not_found_words = self.evaluate(word_vector, word_pairs)

            res = [[test_file]]
            res = res + coeffs + [["not_found_pairs", not_found_pairs],
                                  ["not_found_words", not_found_words],
                                  ["size_data", len(word_pairs), ]]
            scores.append(res)

            print("    > Cantidad de pares con palabras no encontradas: " + str(not_found_pairs) + "\n\n")

        # Guardando resultados
        self.saveResults(word_vector_name, scores)

        print(">>> Resultados")
        for tuple in scores:
            print(tuple)

        del word_vector


    """
    Evaluacion de word embeddings guardados en carpeta embedding.
    """
    def evaluateSavedEmbeddings(self):
        # Creacion de dataset de interseccion, segun embedding en carpeta
        if self._use_intersect_dataset:
            self.createIntersectDataset()

        # Realizacion de test por cada embedding
        print("\n>>> Inicio de test <<<\n")
        for embedding_name in self._embeddings_name_list:
            word_vector_name = embedding_name.split('.')[0]
            word_vector = get_wordvector(embedding_name, self._embeddings_size)

            # Evaluamos embeddings
            self.evaluateWordVector(word_vector_name, word_vector)
