from gensim.models.keyedvectors import KeyedVectors

import shutil
import os
import io
import numpy as np

import Constant

# Archivos que no se pueden usar para relizar test en ambas direcciones de la relacion.
RESTRICTED_FILES = [
    "_español_E01 [pais - capital].txt",
    "_español_E02 [pais - idioma].txt",
    "_español_E04 [nombre - nacionalidad].txt",
    "_español_E05 [nombre - ocupacion].txt",
    "_español_E11 [ciudad_Chile - provincia_Chile].txt"
    "_español_L07 [sinonimos - intensidad].txt",
    "_español_L08 [sinonimos - exacto].txt",
    "_español_L09 [antonimos - grado].txt",
    "_español_L10 [antonimos - binario].txt",
]

_DATASET = Constant.DATA_FOLDER / "AnalogyDataset"
_RESULT = Constant.RESULTS_FOLDER / "Analogy"
_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "Analogy"

# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file
    print(">>> Cargando vectores" + file + "...", end='')
    word_vector = KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)
    print("listo.\n")

    return word_vector

class AnalogyTestClass:
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
    # METRICAS
    ###########################################################################################


    """
    Retorna 1 o 0 si se logra determinar la palabra d, dentro de relacion a:b = c:d
    utilizando 3CosMul como funcion de similaridad
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: 1 o 0, dependiendo si se deduce alguna palabra d 
    """
    def getCoseneSimilarCosmul(self, embedding, p1, p2, q1, q2):
        for a in p1:
            for b in p2:
                for c in q1:
                    res = embedding.most_similar_cosmul(positive=[b, c], negative=[a])
                    res1 = res[0][0]
                    res5 = res[:5]

                    if res1 in q2:
                        return 1

        return 0


    """
    Retorna 1 o 0 si se logra determinar la palabra d, dentro de relacion a:b = c:d
    utilizando 3CosAdd como funcion de similaridad
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: 1 o 0, dependiendo si se deduce alguna palabra d 
    """
    def getCoseneSimilar(self, embedding, p1, p2, q1, q2):
        for a in p1:
            for b in p2:
                for c in q1:
                    res = embedding.most_similar(positive=[b, c], negative=[a])
                    res1 = res[0][0]
                    res5 = res[:5]

                    if res1 in q2:
                        return 1

        return 0


    """
    Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia coseno como funcion de puntaje
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: valor de distancia conseno entre vectores (b-a) y (d-c)
    """
    def getCos(self, embedding, p1, p2, q1, q2):
        result = -1.0
        for a in p1:
            for b in p2:
                for c in q1:
                    for d in q2:
                        r = -1.0
                        a_vec = embedding[a]
                        b_vec = embedding[b]
                        c_vec = embedding[c]
                        d_vec = embedding[d]

                        r = np.dot(b_vec - a_vec, d_vec - c_vec) / (np.linalg.norm(b_vec - a_vec)*np.linalg.norm(d_vec - c_vec))

                        if r > result:
                            result = r

        return result


    """
    Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia euclidiana como funcion de puntaje
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: valor de distancia euclidian entre vectores (b-a) y (d-c)
    """
    def getEuc(self, embedding, p1,p2, q1, q2):
        result = -1.0
        for a in p1:
            for b in p2:
                for c in q1:
                    for d in q2:
                        a_vec = embedding[a]
                        b_vec = embedding[b]
                        c_vec = embedding[c]
                        d_vec = embedding[d]

                        r = 1 - (np.linalg.norm((b_vec - a_vec) - (d_vec - c_vec)) / (np.linalg.norm(b_vec - a_vec) + np.linalg.norm(d_vec - c_vec)))

                        if r > result:
                            result = r

        return result


    """
    Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia coseno como funcion de puntaje, en este
    caso, los vectores son unitarios
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: valor de distancia conseno entre vectores (b-a) y (d-c), con vectores normalizados
    """
    def getNCos(self, embedding, p1,p2, q1, q2):
        return -1.0


    """
    Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia euclidiana como funcion de puntaje, en este
    caso, los vectores son unitarios
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: valor de distancia euclidian entre vectores (b-a) y (d-c), con vectores normalizados
    """
    def getNEuc(self, embedding, p1,p2, q1, q2):
        return 0


    #def getPairDist(embedding, p1,p2, q1, q2):
    #    return 0


    ###########################################################################################
    # MANEJO DE ARCHIVOS Y DATASET
    ###########################################################################################


    """
    Obtencion de nombre de los distintos archivos de test de analogias
    
    :return: lista con path completo de los distintos archivos con pares de palabras para test de analogias
    """
    def getTestFiles(self):
        dataset_folder = _DATASET

        if not dataset_folder.exists():
            raise Exception("No se logro encontrar carpeta con test")

        test_files = []

        for file_name in os.listdir(dataset_folder):
            test_files.append(dataset_folder / file_name)

        return test_files


    """
    Obtencion de path completo hacia los dintintos archivos de test de analogias que no hayan sido evaluados aun
    
    :param test_files: nombre de los archivos que contienen los pares de palabras
    :param embedding_name: nombre del embedding que se va a evaluar
    
    :return: path completo a los archivos con pares de palabras
    """
    def getUntestedFiles(self, embedding_name):
        test_files = self.getTestFiles()
        temp_result_path = _TEMP_RESULT

        # Revisar que existe la carpeta de resultados parciales
        if not temp_result_path.exists():
            return test_files

        # Eliminar archivos que ya han sido utilizados en evaluacion
        test_files_list = []
        for file in test_files:
            # Path hacia el resultado del test asociado al archivo file
            temp_result_file_path = temp_result_path / embedding_name / file.name

            if not temp_result_file_path.exists():
                test_files_list.append(file)

        return test_files_list


    """
    Obtencion de los pares de palabras (a:b) presentes en un archivo de test
    
    :param test_file_path: path hacia archivo con pares de palabras
    :param lower: determina si las palabras solo se consideran en minusculas
    
    :return: lista de pares de palabras 
    """
    def getAnalogyPairs(self, test_file_path):
        if not test_file_path.exists():
            raise Exception("No existe archivo pedido")

        word_pair = []
        with io.open(test_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.lower()

                pair = line.split()
                word_pair.append(pair)

        return word_pair


    '''
    Eliminacion de palabras oov
    
    :param embedding: lista con vectores de palabras
    :param p1: lista de palabras que representa el elemento a en analogia
    :param p2: lista de palabras que representa el elemento b en analogia
    :param q1: lista de palabras que representa el elemento c en analogia
    :param q2: lista de palabras que representa el elemento d en analogia
    
    :return: tupla con lista de palabras, cada lista representa un elemento en la analogia
    '''
    def delete_oov(self, embedding, p1, p2, q1, q2):
        r1 = []
        r2 = []
        t1 = []
        t2 = []

        for x in p1:
            if x in embedding:
                r1.append(x)

        for x in p2:
            if x in embedding:
                r2.append(x)

        for x in q1:
            if x in embedding:
                t1.append(x)

        for x in q2:
            if x in embedding:
                t2.append(x)

        return r1, r2, t1, t2


    ###########################################################################################
    # GUARDAR RESULTADOS
    ###########################################################################################


    """
    Guarda resultados de analogias de forma temporal
    
    :param embedding_name: nombre del embedding que se testeo
    :param test_file_name: nombre del archivo que contiene los pares de palabras usados en el test
    :param results_list: resultados del test sobre distintas metricas, pares (nombre test, resultado)
    """
    def saveTempResults(self, embedding_name, test_file_name, results_list):
        temp_result_embedding = _TEMP_RESULT / embedding_name

        if not temp_result_embedding.exists():
            os.makedirs(temp_result_embedding)

        temp_result_file = temp_result_embedding / test_file_name

        with io.open(temp_result_file, 'w') as f:
            for result in results_list:
                f.write(result[0] + " " + str(result[1]) + "\n")


    """
    Junta todos los resultados de un embedding y luego los guarda en un mismo archivo de resultados.
    
    :param embedding_name: nombre del embedding testeado
    
    :return: resultado de cada uno de los archivos de test, ademas de informacion relacionada a la cantidad de analogias no evaluadas
    """
    def saveResults(self, embedding_name):
        temp_analogy_results_folder = _TEMP_RESULT
        temp_result_embedding = temp_analogy_results_folder / embedding_name


        # Revisar que existe la carpeta de resultado temporales
        if not temp_result_embedding.exists():
            raise Exception("Falta carpeta con resultados temporales, no se pueden obtener resultados")


        # Extraccion de resultados temporales
        test_result_list = os.listdir(temp_result_embedding)
        results = []
        for test_file_name in test_result_list:
            test_result_file = temp_result_embedding / test_file_name

            aux_result = []
            with io.open(test_result_file, 'r') as f:
                for line in f:
                    aux_result.append(line.strip().split())

            results.append([test_file_name, aux_result])

        print(">>> Resultados")
        for r in results:
            print("    ", end='')
            print(r)

        print('')


        # Revisar existencia de capeta donde dejar resultados
        analogy_results_folder = _RESULT
        if not analogy_results_folder.exists():
            os.makedirs(analogy_results_folder)


        # Escribir resultados
        embedding_results = analogy_results_folder / (embedding_name + ".txt")
        with io.open(embedding_results, 'w') as f:
            for r in results:
                f.write(r[0] + "\n")

                for pair_result in r[1]:
                    f.write(pair_result[0] + " " + str(pair_result[1]) + "\n")

        shutil.rmtree(temp_result_embedding)

        return results


    ###########################################################################################
    # EVALUACION POR ANALOGIAS
    ###########################################################################################


    """
    Entrega resultados del test de analogias, utilizando diversas metricas.
    
    :param embedding: 
    :param p1: lista de palabras que representa el elemento a en analogia
    :param p2: lista de palabras que representa el elemento b en analogia
    :param q1: lista de palabras que representa el elemento c en analogia
    :param q2: lista de palabras que representa el elemento d en analogia
    :param all_score: define si se realizan todas las metricas o solo similaridad coseno
    :param all_combination: define si se evaluaran todas las combinaciones posibles de relaciones (3CosAdd, 3CosMul, PairDir)
    
    :return: par de elementos, el primer elemento es una lista con las distintas metricas con las cuales se evalua la analogia,
             el segundo elemento determina si la analogia no fue evaluada, producto de palabras oov
    """
    def evaluateAnalogy(self, embedding, test_file_name, p1, p2, q1, q2, all_score, all_combination):
        # Inicializando variables de resultados
        results = []

        # Eliminamos palabras fuera del vocabulario
        # TODO: palabras fuera del vocabulario se les asigna un vector aleatorio
        p1, p2, q1, q2 = self.delete_oov(embedding, p1, p2, q1, q2)
        if len(p1) < 1 or len(p2) < 1 or len(q1) < 1 or len(q2) < 1:
            return results, True

        # Evaluacion con imilaridad 3CosAdd
        sim_cos_add = 0

        sim_cos_add += self.getCoseneSimilar(embedding, p1, p2, q1, q2)
        if all_combination:
            sim_cos_add += self.getCoseneSimilar(embedding, q1, q2, p1, p2)

            # Algunas relaciones pueden no ser biyectivas
            if not test_file_name in RESTRICTED_FILES:
                sim_cos_add += self.getCoseneSimilar(embedding, p2, p1, q2, q1)
                sim_cos_add += self.getCoseneSimilar(embedding, q2, q1, p2, p1)

        results.append(sim_cos_add)

        if not all_score:
            results = results + (6*[0])

        else:
            # Similaridad 3CosMul
            sim_cos_mul = 0

            sim_cos_mul += self.getCoseneSimilarCosmul(embedding, p1, p2, q1, q2)
            if all_combination:
                sim_cos_mul += self.getCoseneSimilarCosmul(embedding, q1, q2, p1, p2)

                # Algunas relaciones pueden no ser biyectivas
                if not test_file_name in RESTRICTED_FILES:
                    sim_cos_mul += self.getCoseneSimilarCosmul(embedding, p2, p1, q2, q1)
                    sim_cos_mul += self.getCoseneSimilarCosmul(embedding, q2, q1, p2, p1)

            results.append(sim_cos_mul)

            results.append(0)

            # Puntaje coseno
            results.append(self.getCos(embedding, p1, p2, q1, q2))

            # Puntaje euclidiano
            results.append(self.getEuc(embedding, p1, p2, q1, q2))

            # Puntaje n-coseno
            results.append(self.getNCos(embedding, p1, p2, q1, q2))

            # Puntaje n-euclidiano
            results.append(self.getNEuc(embedding, p1, p2, q1, q2))

        return results, False


    """
    Realizacion de test de analogias
    
    :param embedding: lista de vectores de palabras
    :param embedding_name: nobre del vector de palabras
    :param all_score: determina si se utilizan todas las metricas disponibles
    :param all_combination: determina si se utlizan todas las posibles combinaciones para una relaciondad,
             considerando que hay relaciones que no son biyectivas
                            
    :return: lista con los resultados, individuales de cada test, con las metricas disponibles
    """
    def analogyTest(self, all_score=False, all_combination=False):
        # Realizacion de test por cada embedding
        for embedding_name in self._embeddings_name_list:
            word_vector_name = embedding_name.split('.')[0]
            print(">>> Cargando vectores " + word_vector_name + "...", end='')
            word_vector = get_wordvector(embedding_name, self._embeddings_size)
            print("listo.\n")

            # Obtencion de archivos que faltan por testear
            test_file_list = self.getUntestedFiles(word_vector_name)

            # Revisamos todos los archivos para realizar test
            # TODO: usar threads o weas uwu
            for file in test_file_list:
                print(">>> Testing: ", end='')
                print(file.name)
                pair_list = self.getAnalogyPairs(file)


                # Inicializamos variables para guardar metricas obtenidas para el archivo de test
                total_test_result = []
                similarity_total = [0, 0, 0]
                che_metric = [0, 0, 0, 0]

                count_relations = 0
                count_multiply = 1

                omited_count = 0


                # En caso de evaluar todas las combinaciones, diferenciamos los test que tienen relaciones no biyectivas
                if all_combination:
                    if not file.name in RESTRICTED_FILES:
                        count_multiply = 4
                    else:
                        count_multiply = 2


                # Evaluamos todas las 4-tuplas posibles a partir de todos los pares presentes en el archivo file
                for i in range(len(pair_list)):
                    for j in range(len(pair_list)):
                        if i == j:
                            continue


                        # Generamos las 4-tuplas (p1, p2, q1, q2), donde "p1 es a p2 como q1 es aq2"
                        p = pair_list[i]
                        q = pair_list[j]

                        p1 = p[0].strip().split('/')
                        p2 = p[1].strip().split('/')
                        q1 = q[0].strip().split('/')
                        q2 = q[1].strip().split('/')

                        """
                        print(p1, end=' : ')
                        print(p2, end=' - ')
                        print(q1, end=' : ')
                        print(q2)
                        """


                        # Obtencion de resultados a partir de las metricas disponibles
                        result_tuple, omited_relation = self.evaluateAnalogy(word_vector, file.name, p1, p2, q1, q2, all_score, all_combination)
                        if omited_relation:
                            omited_count += 1
                            continue


                        # Contar la cantidad de relaciones que se pueden hacer con todas las tuplas posibles
                        count_relations += 1

                        # Separamos los resultados por:
                        # -> Similaridad
                        similarity_total[0] += result_tuple[0]
                        similarity_total[1] += result_tuple[1]
                        similarity_total[2] += result_tuple[2]

                        # -> Puntajes definido por Che
                        che_metric[0] += result_tuple[3]
                        che_metric[1] += result_tuple[4]
                        che_metric[2] += result_tuple[5]
                        che_metric[3] += result_tuple[6]


                # Calculamos los resultados totales del test
                # Similaridad
                total_test_result.append(["3CosAdd", similarity_total[0] * 1.0 / (1.0 * count_relations * count_multiply) if count_relations > 0 else "Nan"])
                total_test_result.append(["3CosMul", similarity_total[1] * 1.0 / (1.0 * count_relations * count_multiply) if count_relations > 0 else "Nan"])
                total_test_result.append(["PairDir", similarity_total[2] * 1.0 / (1.0 * count_relations * count_multiply) if count_relations > 0 else "Nan"])

                # Puntajes definido por Che
                total_test_result.append(["cos", che_metric[0] * 1.0 / (1.0 * count_relations) if count_relations > 0 else "Nan"])
                total_test_result.append(["euc", che_metric[1] * 1.0 / (1.0 * count_relations) if count_relations > 0 else "Nan"])
                total_test_result.append(["ncos", che_metric[2] * 1.0 / (1.0 * count_relations) if count_relations > 0 else "Nan"])
                total_test_result.append(["neuc", che_metric[3] * 1.0 / (1.0 * count_relations) if count_relations > 0 else "Nan"])

                # Cantidad de pares de relaciones omitidas
                total_test_result.append(["omited", omited_count])
                total_test_result.append(["%omited", omited_count / (omited_count + count_relations)])

                # Guardamos los resultados de forma temporal
                self.saveTempResults(word_vector_name, file.name, total_test_result)

            self.saveResults(word_vector_name)

