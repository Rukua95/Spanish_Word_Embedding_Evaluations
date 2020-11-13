from gensim.models.keyedvectors import KeyedVectors

import numpy as np

import shutil
import os
import io
import Constant

# Clase para realizar evaluacion de word embedding segun outlier detection
class OutlierDetectionTestClass:

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "OutlierDetectionDataset"

    _INTERSECT_DATASET = Constant.DATA_FOLDER / "_intersection_OutlierDetectionDataset"
    _ORIGINAL_DATASET = Constant.DATA_FOLDER / "OutlierDetectionDataset"

    _RESULT = Constant.RESULTS_FOLDER / "OutlierDetection"

    def __init__(self, lower=True, use_intersect_dataset=False):
        #print(">>> Test de Outlier Detection <<<")

        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset

        # Interseccion de datasets
        if self._use_intersect_dataset:
            self._DATASET = self._INTERSECT_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "_intersection_OutlierDetection"

            self.createIntersectDataset()

        else:
            self._DATASET = self._ORIGINAL_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "OutlierDetection"

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
            u = embedding[word]
            v = embedding[w]

            sum += np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
            sum += np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v))
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
        #print(">>> Eliminando terminos de mas de una palabra")
        main_oov = 0
        outlier_oov = 0
        filter_main_set = []
        filter_outlier_set = []

        for w in main_set:
            if len(w.strip().split('_')) > 1 or (w not in embedding):
                main_oov += 1
                continue

            filter_main_set.append(w)

        for w in outlier_set:
            if len(w.strip().split('_')) > 1 or (w not in embedding):
                outlier_oov += 1
                continue

            filter_outlier_set.append(w)

        return filter_main_set, filter_outlier_set, main_oov, outlier_oov


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
            # Obtenemos puntaje del resto de elementos en el conjunto
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

        # Obtencion de porcentaje de palabras oov
        total_main = 0
        total_outlier = 0
        total_main_oov = 0
        total_outlier_oov = 0
        omited_test = 0

        count = 0
        for test in test_sets:
            count += 1
            #print(" > Test", str(count), "de", str(len(test_sets)))


            # Conjunto principal y outlier
            main_set, outlier_set = test
            #print("   > Sets originales:")
            #print("    ", main_set)
            #print("    ", outlier_set)


            # Cuenta cantidad de palabras oov y elimina terminos que utilicen mas de una palabra
            main_set, outlier_set, main_oov, outlier_oov = self.omitOOVWord(embedding, main_set, outlier_set)

            #print(" > Sets editados:")
            #print("    ", main_set, "- oov words:", str(main_oov), "de", str(len(main_set)))
            #print("    ", outlier_set, "- oov words:", str(outlier_oov), "de", str(len(outlier_set)))

            if len(main_set) < 2 or len(outlier_set) < 1:
                #print(" > Test set no cumple con condiciones de evaluacion, se procede a omitir")
                omited_test += 1
                continue

            total_main += len(main_set)
            total_outlier += len(outlier_set)
            total_main_oov += main_oov
            total_outlier_oov += outlier_oov


            # Obtencion de listas OP y OD
            OP_list, OD_list = self.getFileScores(embedding, main_set, outlier_set)

            #print(" > OP:", OP_list)
            #print(" > OD:", OD_list)

            sum_op += (sum(OP_list) / len(main_set))
            sum_od += sum(OD_list)

            cant_test += len(outlier_set)

        results = []

        if cant_test == 0:
            results = ["Nan", "Nan", "Nan", "Nan", "Nan"]
        else:
            results = [
                (sum_op / cant_test),
                (sum_od / cant_test),
                (total_main_oov / total_main),
                (total_outlier_oov / total_outlier),
                omited_test,
            ]

        return results


    ###########################################################################################
    # MANEJO DE ARCHIVOS Y DATASET
    ###########################################################################################

    def resetIntersectDataset(self):
        #print("Eliminando archivos en carpeta de interseccion de dataset")
        intersect_dataset_path = self._INTERSECT_DATASET
        if intersect_dataset_path.exists():
            shutil.rmtree(intersect_dataset_path)

        if self._use_intersect_dataset:
            self.createIntersectDataset()

    """
    Metodo que crea dataset con la interseccion de vocabulario de los embeddings en carpeta
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
    Metodo que elimina palabras de los dataset, que esten fuera del vocabulario del word embeddings dado

    :param word_vector: word embedding para comparar vocabulario
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
            main_set_lines = []
            outlier_set_lines = []

            # Revisar el dataset intersectado que llevamos hasta el momento
            with io.open(file_path, 'r', encoding='utf-8') as f:
                for line in f:

                    # Verificar que se lea separacion entre main y outlier set
                    if line == "\n":
                        main_set_lines = outlier_set_lines
                        outlier_set_lines = []
                        continue

                    tupla = line.lower().split() if self._lower else line.split()

                    # Revisar cantidad de palabras
                    if len(tupla) > 1:
                        deleted_element += 1
                        continue

                    # Revisar que palabra esta en vocabulario
                    if tupla[0] not in word_vector:
                        deleted_element += 1
                        continue

                    outlier_set_lines.append(line)

            total_lines = len(main_set_lines) + len(outlier_set_lines) + deleted_element
            #print("   > Lineas eliminadas:", str(deleted_element), "de", str(total_lines))

            # Eliminamos archivo que no aporta al analisis
            if len(main_set_lines) < 2 or len(outlier_set_lines) < 1:
                deleted_files += 1
                to_delete_files.append(file_path)
                #print("   > Conjunto principal u outlier vacios, se procede a eliminar")
                continue

            # Escribimos documento
            with io.open(file_path, 'w', encoding='utf-8') as f:
                for line in main_set_lines:
                    f.write(line)

                f.write("\n")

                for line in outlier_set_lines:
                    f.write(line)

        #print(" > Archivos a eliminar: " + str(deleted_files) + "\n")
        for file in to_delete_files:
            os.remove(file)

        return True if len(os.listdir(next_dataset_path)) > 0 else False


    """
    Obtencion de la lista de archivos test
    
    :return: lista con nombre de archivos con test de outlier detection
    """
    def getTestFiles(self):
        #print(" > Obteniendo nombre de archivos de test desde:", str(self._DATASET))
        if not self._DATASET.exists():
            raise Exception("No se logro encontrar carpeta con test")

        return os.listdir(self._DATASET)


    """
    Obtencion de las palabras desde el archivo de test, palabras del conjunto y conunto outlier
    
    :param file_name: nombre de archivo con test
    :param lower: determina si se utilizan solo minusculas en el test
    
    :return: par de conjuntos, conjunto principal y conjunto outlier
    """
    def getWords(self, file_name):
        main_set = []
        outlier_set = []
        with io.open(self._DATASET / file_name, 'r', encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    main_set = outlier_set
                    outlier_set = []
                    continue

                line = line.strip()
                line = line.lower() if self._lower else line

                outlier_set.append(line)

        return main_set, outlier_set


    """
    Metodo para la obtencion de todos los conjuntos de palabras que se utilizaran como test
    
    :param lower: determina si se utilizan solo minusculas en el test
    
    :return: lista de pares de conjuntos, conjunto principal y outlier, de cada test
    """
    def getTests(self):
        #print(" > Extraccion de datasets")

        file_list = self.getTestFiles()
        test_list = []

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
        save_path = self._RESULT

        if not save_path.exists():
            os.makedirs(save_path)

        save_path = self._RESULT / (embedding_name + ".txt")

        with io.open(save_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(r[0] + " " + str(r[1]) + "\n")

    ###########################################################################################
    # EVALUACION POR OUTLIER DETECTION
    ###########################################################################################

    """
    Evalua un word embedding especifico y guarda el resultado en carpeta de resultados

    :param word_vector_name: nombre de word embedding, utilizado para especificar nombre de
                             archivo de resultados
    :param word_vector: word embedding a evaluar
    """
    def evaluateWordVector(self, word_vector_name, word_vector):
        #print(">>> Evaluando embedding ", str(word_vector_name))

        # Obtencion de conjuntos, principal y outlier
        test_list = self.getTests()

        # Obtencion y limpieza de resultados
        word_vector_results = self.getScores(word_vector, test_list)
        word_vector_results = [
            ["accuraccy", word_vector_results[0]],
            ["OPP", word_vector_results[1]],
            ["%_main_oov", word_vector_results[2]],
            ["%_outlier_oov", word_vector_results[3]],
            ["omited sets", word_vector_results[4]]
        ]

        # Guardado de resultados
        self.saveResults(word_vector_name, word_vector_results)

        return word_vector_results
