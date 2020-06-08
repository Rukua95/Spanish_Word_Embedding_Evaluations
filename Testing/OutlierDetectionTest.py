from pathlib import Path

from gensim.models.keyedvectors import KeyedVectors

import os
import io
import numpy as np

import Constant


_DATASET = Constant.DATA_FOLDER / "OutlierDetectionDataset"
_RESULT = Constant.RESULTS_FOLDER / "OutlierDetection"
_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "OutlierDetection"

# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file

    return KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)

###########################################################################################
# PUNTUACION
###########################################################################################
class OutlierDetectionTestClass:
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _lower = True
    _use_intersect_dataset = False
    _oov_word = {}

    def __init__(self, cantidad=None, lower=True, use_intersect_dataset=False):
        print("Test de Outlier Detection")

        self._embeddings_size = cantidad
        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset

    """
    Retorna Pseudo-Inverted Compactness Score (Camacho-Colados) para todos los w en el conjunto C = W + {w}
    
    :param embedding: lista de vectores de palabras
    :param W: lista de palabras
    :param w: palabra a la cual calcula puntaje respecto a conjunto W
    
    :return: puntaje de palabra w respecto a conjunto W
    """
    def getPseudoInvertedCompactnessScore(self, embedding, W, w):
        sum = 0
        k = 0
        for word in W:
            u = np.random.rand(embedding.vector_size)
            v = np.random.rand(embedding.vector_size)

            if word in embedding:
                u = embedding[word]
            else:
                if word not in self._oov_word.keys():
                    self._oov_word[word] = np.random.rand(embedding.vector_size)

                u = self._oov_word[word]

            if w in embedding:
                v = embedding[w]
            else:
                if w not in self._oov_word.keys():
                    self._oov_word[w] = np.random.rand(embedding.vector_size)

                v = self._oov_word[w]

            sum += u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
            sum += v.dot(u) / (np.linalg.norm(u) * np.linalg.norm(v))
            k += 1

        return (sum / k)


    """
    Elimina palabras dentro de conjunto principal y conjunto outlier, que no aparezcan en el vocabulario
    
    :param embedding: lista de vectores de palabras
    :param main_set: lista de palabras del conjunto principal
    :param outlier_set: lista de palabras del conjunto outlier
    
    :return: la cantidad de palabras omitidas
    """
    def omitOOVWord(self, embedding, main_set, outlier_set):
        main_oov = 0
        outlier_oov = 0

        for w in main_set:
            if w not in embedding:
                main_oov += 1

        for w in outlier_set:
            if w not in embedding:
                outlier_oov += 1

        return main_oov, outlier_oov


    """
    Obtencion de puntaje del test para un solo archivo/conjunto
    
    :param embedding: lista de vectores de palabras
    :param main_set: conjunto principal de palabras
    :param outlier_set: conjunto de palabras outliers
    :param exist_oov: determina si se solo palabras en la interseccion de vocabularios
    
    :return: valor de OP y OD para cada palabra en outlier_set, respecto a main_set
    """
    def getFileScores(self, embedding, main_set, outlier_set):
        OP = []
        OD = []
        for outlier in outlier_set:
            # Obtencion de puntaje para outlier (nos importa su posicion respecto al resto de puntajes)
            W = main_set
            p = self.getPseudoInvertedCompactnessScore(embedding, W, outlier)
            pos = len(main_set)

            W_outlier = W
            for i in range(len(W_outlier)):

                C = W_outlier[0: i] + W_outlier[i+1:] + [outlier]
                w = W_outlier[i]

                p_i = self.getPseudoInvertedCompactnessScore(embedding, C, w)
                if p_i < p:
                    pos -= 1

            OP.append(pos)
            OD.append(1 if pos == len(main_set) else 0)

        return OP, OD


    """
    Obtencion de accuracy y OPP
    
    :param embedding: lista de vectores de palabras
    :param test_sets: lista de pares de conjuntos [main_set, outlier_set]
    :param exist_oov: determina si pueden existir palabras fuera del vocabulario
    
    :return: accuracy y OPP, e informacion relacionada a palabras oov
    """
    def getScores(self, embedding, test_sets):
        # Suma de valores op y od
        cant_test = 0
        sum_op = 0.0
        sum_od = 0.0

        # Obtencion de porcentaje de omision
        total_main = 0
        total_main_oov = 0
        total_outlier = 0
        total_outlier_oov = 0

        count = 0
        for test in test_sets:
            count += 1
            print(">>> Test " + str(count) + " of " + str(len(test_sets)))


            # Conjunto principal y outlier
            main_set, outlier_set = test
            total_main += len(main_set)
            total_outlier += len(outlier_set)

            print(">>> Original set:", end='\n    ')
            print(main_set, end='\n    ')
            print(outlier_set)


            # Determinar sets con palabras omitidas
            main_oov, outlier_oov = self.omitOOVWord(embedding, main_set, outlier_set)
            total_main_oov += main_oov
            total_outlier_oov += outlier_oov


            # Obtencion de listas OP y OD
            OP_list, OD_list = self.getFileScores(embedding, main_set, outlier_set)

            print("OP and OD list:")
            print(OP_list)
            print(OD_list)

            sum_op += (sum(OP_list) / len(main_set))
            sum_od += sum(OD_list)

            cant_test += len(outlier_set)

        results = []

        if cant_test == 0:
            results = ["Nan", "Nan", "Nan", "Nan"]
        else:
            results = [
                (sum_op / cant_test),
                (sum_od / cant_test),
                (total_main_oov / total_main),
                (total_outlier_oov / total_outlier)
            ]

        return results


    ###########################################################################################
    # MANEJO DE ARCHIVOS Y DATASET
    ###########################################################################################

    def intersectDataset(self, word_vector):
        # TODO: revisar si hay archivos en la carpeta de interseccion
        # TODO: realizar interseccion de dataset con dataset de interseccion o dataset original
        pass


    """
    Obtencion de la lista de archivos test
    
    :return: lista con nombre de archivos con test de outlier detection
    """
    def getTestFiles(self):
        if not _DATASET.exists():
            raise Exception("No se logro encontrar carpeta con test")

        return os.listdir(_DATASET)


    """
    Obtencion de las palabras desde el archivo de test, palabras del conjunto y conunto outlier
    
    :param file_name: nombre de archivo con test
    :param lower: determina si se utilizan solo minusculas en el test
    
    :return: par de conjuntos, conjunto principal y conjunto outlier
    """
    def getWords(self, file_name):
        main_set = []
        outlier_set = []

        with io.open(_DATASET / file_name, 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    main_set = outlier_set
                    outlier_set = []
                    continue

                line = line.strip()
                line = line.lower()

                outlier_set.append(line)

        return main_set, outlier_set


    """
    Metodo para la obtencion de todos los conjuntos de palabras que se utilizaran como test
    
    :param lower: determina si se utilizan solo minusculas en el test
    
    :return: lista de pares de conjuntos, conjunto principal y outlier, de cada test
    """
    def getTests(self):
        file_list = self.getTestFiles()
        test_list = []
        count = 0

        for file in file_list:
            test_list.append(self.getWords(file))

        return test_list


    ###########################################################################################
    # GUARDAR RESULTADOS
    ###########################################################################################

    """
    Guardado de resultados del test por outlier detection
    
    :param embedding_name: nombre de embedding evaluado
    :param results: lista de los resultados (accuraccy, OPP) y datos (% oov outlier, % ovv main set, % grupos omitidos)
                    de un embedding, la lista se compone de pares [nombre_resultado, valor_resultado]
    :param exist_oov: determina si se consideran palabras fuera del vocabulario
    """
    def saveResults(self, embedding_name, results):
        extension = ""

        save_path = _RESULT
        if not save_path.exists():
            os.makedirs(save_path)

        save_path = _RESULT / (embedding_name + extension + ".txt")

        with io.open(save_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(r[0] + " " + str(r[1]) + "\n")


    ###########################################################################################
    # EVALUACION POR OUTLIER DETECTION
    ###########################################################################################


    """
    Realizacion de test de outlier detection
    
    :param embedding: lista de vectores de palabras
    :param embedding_name: nombre del embedding a evaluar
    :param existe_ovv: determina si se consideran palabras fuera del vocabulario
    
    :return: lista con resultados de accuraccy y OPP, ademas de info sobre palabras oov
    """
    def outlierDetectionTest(self):
        results = {}

        # Realizacion de test por cada embedding
        for embedding_name in self._embeddings_name_list:
            word_vector_name = embedding_name.split('.')[0]
            print(">>> Cargando vectores " + word_vector_name + "...", end='')
            word_vector = get_wordvector(embedding_name, self._embeddings_size)
            print("listo.\n")


            # Obtencion de conjuntos, principal y outlier
            test_list = self.getTests()


            # Obtencion y limpieza de resultados
            word_vector_results = self.getScores(word_vector, test_list)
            word_vector_results = [
                ["accuraccy", word_vector_results[0]],
                ["OPP", word_vector_results[1]],
                ["%_main_oov", word_vector_results[2]],
                ["%_outlier_oov", word_vector_results[3]],
            ]

            print(">>> Resultados:\n    ", end='')
            print(word_vector_results, end='\n\n')


            # Guardado de resultados
            self.saveResults(word_vector_name, word_vector_results)
            results[word_vector_name] = word_vector_results

        return results

