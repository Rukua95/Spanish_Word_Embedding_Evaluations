from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import spearmanr, kendalltau, pearsonr

import numpy as np

import shutil
import os
import io
import Constant

# Clase para realizar evaluacion de word embedding segun similaridad semantica.
class SimilarityTestClass:

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "SimilarityDataset"

    _INTERSECT_DATASET = Constant.DATA_FOLDER / "_intersection_SimilarityDataset"
    _ORIGINAL_DATASET = Constant.DATA_FOLDER / "SimilarityDataset"

    _RESULT = Constant.RESULTS_FOLDER / "Similarity"

    """
    Constructor de la clase para evaluacion de word embeddings segun similaridad de palabras
    
    :param lower: booleano para determinar como se leen las minusculas en los dataset utilizado, por defecto las 
                  mayusculas son reemplazadas por minusculas
    :param use_intersect_dataset: booleano que determinar si se utiliza el vocabulario de los dataset intersectados
                                  con el vocabulario de los embeddings, por defecto es False
    :param debug: booleano para proceso de debugging
    :param dataset: lista con el nombre de los datasets a utilizar, si se encuentra vacia, se utilizan todos los 
                    datasets disponibles
    """
    def __init__(self, lower=True, use_intersect_dataset=False, datasets=[]):
        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset
        self._datasets = datasets

        # Interseccion de datasets
        if self._use_intersect_dataset:
            self._DATASET = self._INTERSECT_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "_intersection_Similarity"

            self.createIntersectDataset()

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
    def evaluate(self, word_vector, word_pairs):
        not_found_pairs = 0
        not_found_words = 0
        repeated_pairs = 0

        not_found_list = []

        pred = []
        gold = []
        reg = []
        for word1, word2, similarity in word_pairs:
            w1 = word1 in word_vector
            w2 = word2 in word_vector

            # Eliminar pares de palabras repetidos
            aux = [word1, word2]
            aux.sort()
            if aux in reg:
                repeated_pairs += 1
                continue

            else:
                reg.append(aux)

            # Eliminar pares de palabras no encontradas
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
                u = word_vector[word1]
                v = word_vector[word2]

            # Calculo de similaridad coseno
            score = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

            gold.append(similarity)
            pred.append(score)

        #print("    Pares repetidos:", repeated_pairs)
        #print("    Pares de palabras eliminadas:", str(not_found_pairs))
        #print("    Palabras no encontradas:", str(not_found_words))
        #print("    ", not_found_list)

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
    Metodo para elimina los actuales datasets generados como la interseccion de los vocabularios de los embeddings,
    y genera nuevos datasets para ser utilizados. Si no se utiliza la interseccion de los vocabularios, este metodo solo
    elimina el dataset de la interseccion de vocabularios.
    """
    def resetIntersectDataset(self):
        #print(">>> Reset/Eliminacion de carpeta de interseccion de datasets")

        intersect_dataset_path = self._INTERSECT_DATASET

        if intersect_dataset_path.exists():
            shutil.rmtree(intersect_dataset_path)

        if self._use_intersect_dataset:
            self.createIntersectDataset()


    """
    Metodo que prepara los dataset utilizados en la evaluacion, para ser intersectados con el vocabulario de embeddings.
    En caso de que existan datasets ya intersectados, este metodo solo prepara los datasets que no se encuentren 
    intersectados.
    """
    def createIntersectDataset(self):
        #print(">>> Copiando dataset original para realizar interseccion")

        # Verificar que existe carpeta para guardar nuevo dataset
        if not self._INTERSECT_DATASET.exists():
            os.makedirs(self._INTERSECT_DATASET)

        # Verificar si hay datasets ya intersectados
        for file_name in os.listdir(self._ORIGINAL_DATASET):
            if file_name not in os.listdir(self._INTERSECT_DATASET):
                origin_file = self._ORIGINAL_DATASET / file_name
                shutil.copy(origin_file, self._INTERSECT_DATASET)
                #print("   ", file_name, " no se encuentra en dataset de interseccion, copiando")


    """
    Metodo que intersecta los datasets utilizados en la evaluacion, con el vocabulario del embedding dado. Previo a 
    utilizar este metodo, es necesario crear los datasets para el proceso de interseccion.
    
    :param word_vector: word embedding para comparar vocabulario
    :return: booleano que determina si despues de realizada la interseccion existen dataset con los cuales realizar
        la evaluacion.
    """
    def intersectDataset(self, word_vector):
        #print(">>> Intersectando datasets con vocabulario de embedding...")

        next_dataset_path = self._INTERSECT_DATASET
        deleted_files = 0

        # Revisar cada archivo dentro de la carpeta de dataset
        #print(" > Revision de archivos en dataset")

        to_delete_files = []
        for file_name in os.listdir(next_dataset_path):
            #print("   Revisando " + file_name)

            file_path = next_dataset_path / file_name
            deleted_element = 0
            lines = []

            # Revisar el dataset intersectado que llevamos hasta el momento
            with io.open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    tupla = line.lower().split() if self._lower else line.split()

                    # Revisar cantidad de palabras en tupla
                    if len(tupla) != 3:
                        deleted_element += 1
                        continue

                    # Revisar que todas las palabras estan en el vocabulario
                    if tupla[0] not in word_vector or tupla[1] not in word_vector or len(tupla) != 3:
                        deleted_element += 1
                        continue

                    lines.append(line)

            #print("   > Lineas eliminadas:", str(deleted_element), "de", str(deleted_element + len(lines)))

            # Eliminamos archivo que no aporta al analisis
            if len(lines) == 0:
                deleted_files += 1
                to_delete_files.append(file_path)
                #print("   > Archivo esta vacio, se procede a eliminar")

                continue

            # Escribir documento
            with io.open(file_path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line)

        #print(" > Archivos a eliminar:", str(deleted_files) + "\n")

        for file in to_delete_files:
            os.remove(file)

        return True if len(os.listdir(next_dataset_path)) > 0 else False


    """
    Obtencion de datasets requeridos para realizar la evaluacion.
    
    :return: lista de nombres de dataset.
    """
    def getTestFiles(self):
        #print(" > Obteniendo nombre de archivos de test desde:", str(self._DATASET))
        if not self._DATASET.exists():
            raise Exception("No se logro encontrar carpeta con test")

        if len(self._datasets) == 0:
            return os.listdir(self._DATASET)
        else:
            datasets_files = []
            for set_file in os.listdir(self._DATASET):
                use = False
                for name in self._datasets:
                    if name in set_file:
                        use = True

                if use:
                    datasets_files.append(set_file)

            if len(datasets_files) == 0:
                raise Exception("Datasets a utilizar no existen")

            return datasets_files


    """
    Obtencion de informacion presente en un dataset dado.
    
    :param file: nombre de archivo del dataset.
    
    :return: lista con pares de palabras y su puntaje de similaridad.
    """
    def getWordPairs(self, file):
        #print(" > Extraccion de dataset")

        word_pairs = []
        with io.open(self._DATASET / file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.lower() if self._lower else line
                line = line.split()

                # Ignorar frases y solo considerar palabras
                if len(line) != 3:
                    assert len(line) > 3
                    continue

                pair = (line[0], line[1], float(line[2]))
                word_pairs.append(pair)

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
        #print(">>> Guardando resultados en:", str(result_path))

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

    :param word_vector_name: nombre de word embedding, utilizado para especificar nombre de 
        archivo de resultados
    :param word_vector: word embedding a evaluar
    
    :return: lista con resultados para cada dataset
    """
    def evaluateWordVector(self, word_vector_name, word_vector):
        #print(">>> Evaluando embedding ", str(word_vector_name))

        # Obtencion de nombre de archivos de test
        test_file_list = self.getTestFiles()
        scores = []

        # Test en archivos individuales
        for test_file in test_file_list:
            #print(" > Evaluacion con dataset ", str(test_file))

            word_pairs = self.getWordPairs(test_file)

            # Evaluamos embeddings con dataset especifico
            coeffs, not_found_pairs, not_found_words = self.evaluate(word_vector, word_pairs)

            res = [[test_file]]
            res = res + coeffs + [["not_found_pairs", not_found_pairs],
                                  ["not_found_words", not_found_words],
                                  ["size_data", len(word_pairs), ]]
            scores.append(res)

        # Guardando resultados
        self.saveResults(word_vector_name, scores)

        #for r in scores:
        #    print(r)

        return scores