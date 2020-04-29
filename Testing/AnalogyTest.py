import os
import io
import shutil

import numpy as np

from pathlib import Path

RESTRICTED_FILES = []
CURRENT_PATH = os.getcwd()
path = Path(CURRENT_PATH)

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
def getCoseneSimilarCosmul(embedding, p1, p2, q1, q2):
    for a in p1:
        for b in p2:
            for c in q1:
                res = ''
                try:
                    res = embedding.most_similar_cosmul(positive=[b, c], negative=[a])
                    res = res[0][0]
                except:
                    res = ''

                if res in q2:
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
def getCoseneSimilar(embedding, p1, p2, q1, q2):
    for a in p1:
        for b in p2:
            for c in q1:
                res = ''
                try:
                    res = embedding.most_similar(positive=[b, c], negative=[a])
                    res = res[0][0]
                except:
                    res = ''

                if res in q2:
                    return 1

    return 0


"""
Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia coseno como funcion de puntaje

:param embedding: embeddings
:param p1: dentro de la relacion, palabras que pueden ser a
:param p2: dentro de la relacion, palabras que pueden ser b
:param q1: dentro de la relacion, palabras que pueden ser c
:param q2: dentro de la relacion, palabras que pueden ser d
:return: 
"""
def getCos(embedding, p1, p2, q1, q2):
    result = -1.0
    for a in p1:
        for b in p2:
            for c in q1:
                for d in q2:
                    r = -1.0

                    try:
                        a_vec = embedding[a]
                        b_vec = embedding[b]
                        c_vec = embedding[c]
                        d_vec = embedding[d]

                        r = np.dot(b_vec - a_vec, d_vec - c_vec) / (np.linalg.norm(b_vec - a_vec)*np.linalg.norm(d_vec - c_vec))

                    except:
                        r = -1.0

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
"""
def getEuc(embedding, p1,p2, q1, q2):
    result = -1.0
    for a in p1:
        for b in p2:
            for c in q1:
                for d in q2:
                    r = 0

                    try:
                        a_vec = embedding[a]
                        b_vec = embedding[b]
                        c_vec = embedding[c]
                        d_vec = embedding[d]

                        r = 1 - (np.linalg.norm((b_vec - a_vec) - (d_vec - c_vec)) / (np.linalg.norm(b_vec - a_vec) + np.linalg.norm(d_vec - c_vec)))

                    except:
                        r = 0

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
:return:
"""
def getNCos(embedding, p1,p2, q1, q2):
    return -1.0

"""
Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia euclidiana como funcion de puntaje, en este
caso, los vectores son unitarios

:param embedding: embeddings
:param p1: dentro de la relacion, palabras que pueden ser a
:param p2: dentro de la relacion, palabras que pueden ser b
:param q1: dentro de la relacion, palabras que pueden ser c
:param q2: dentro de la relacion, palabras que pueden ser d
:return:
"""
def getNEuc(embedding, p1,p2, q1, q2):
    return 0


def getPairDist(embedding, p1,p2, q1, q2):
    pass


###########################################################################################
# FILES HANDLING
###########################################################################################

"""
Obtencion de nombre de los distintos archivos de test de analogias
:return: lista con path completo de los distintos archivos con pares de palabras para test de analogias
"""
def getTestFiles():
    dataset_folder = path.parent / "Datasets/AnalogyDataset"

    if not dataset_folder.exists():
        raise Exception('Dataset folder not found')

    test_files = []
    sub_dataset_folder = os.listdir(dataset_folder)
    for folder in sub_dataset_folder:
        # Hay distintos subcarpetas para los test de analogias.
        # Inflexion, Derivacion, Enciclopedia, Lexicografico.
        sub_folder = dataset_folder / folder

        test_files = test_files + list(map(lambda f: sub_folder / f, os.listdir(sub_folder)))

    return test_files


"""
Obtencion de path completo hacia los dintintos archivos de test de analogias que no hayan sido evaluados aun

:param test_files: nombre de los archivos que contienen los pares de palabras
:param embedding_name: nombre del embedding que se va a evaluar
:return: path completo a los archivos con pares de palabras
"""
def getUntestedFiles(test_files, embedding_name):
    temp_result_path = path.parent / "TempResults/Analogy"

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


###########################################################################################
# SAVE RESULTS
###########################################################################################

"""
Guarda resultados de analogias de forma temporal

:param embedding_name: nombre del embedding que se testeo
:param test_file_name: nombre del archivo que contiene los pares de palabras usados en el test
:param results_list: resultados del test sobre distintas metricas, pares (nombre test, resultado)
"""
def saveTempResults(embedding_name, test_file_name, results_list):
    temp_analogy_results_folder = path.parent / "TempResults/Analogy"
    temp_result_embedding = temp_analogy_results_folder / embedding_name

    if not temp_result_embedding.exists():
        os.makedirs(temp_result_embedding)

    temp_result_file = temp_result_embedding / test_file_name

    with io.open(temp_result_file, 'w') as f:
        for result in results_list:
            f.write(result[0] + " " + str(result[1]) + "\n")


"""
Junta todos los resultados de un embedding y luego los guarda en un mismo archivo de resultados.

:param embedding_name: nombre del embedding testeado
"""
def saveResults(embedding_name):
    temp_analogy_results_folder = path.parent / "TempResults/Analogy"
    temp_result_embedding = temp_analogy_results_folder / embedding_name

    # Revisar que existe la carpeta de resultado temporales
    if not temp_result_embedding.exists():
        raise Exception("Falta carpeta con resultados temporales")

    test_result_list = os.listdir(temp_result_embedding)
    results = []
    for test_file_name in test_result_list:
        test_result_file = temp_result_embedding / test_file_name

        aux_result = []
        with io.open(test_result_file, 'r') as f:
            for line in f:
                aux_result.append(line.strip().split())

        results.append([test_file_name, aux_result])

    for r in results:
        print(r)

    analogy_results_folder = path.parent / "Resultados/Analogy"
    if not analogy_results_folder.exists():
        os.makedirs(analogy_results_folder)

    embedding_results = analogy_results_folder / (embedding_name + ".txt")
    with io.open(embedding_results, 'w') as f:
        for r in results:
            f.write(r[0] + "\n")

            for pair_result in r[1]:
                f.write(pair_result[0] + " " + str(pair_result[1]) + "\n")

    shutil.rmtree(temp_result_embedding)


###########################################################################################
# ANALOGY EVALUATION
###########################################################################################


"""
Obtencion de los pares de palabras (a:b) presentes en un archivo de test
"""
def getAnalogyPairs(test_file_path):
    if not test_file_path.exists():
        raise Exception("No existe archivo pedido")

    word_pair = []
    with io.open(test_file_path, 'r') as f:
        for line in f:
            pair = line.strip().split()
            word_pair.append(pair)

    return word_pair


"""
Entrega resultados del test de analogias, utilizando diversas metricas.
:param embedding: 
:param p1:
:param p2:
:param q1:
:param q2:
:param all_score: define si se realizan todas las metricas o solo similaridad coseno
:param all_combination: define si se evaluaran todas las combinaciones posibles de relaciones (3CosAdd, 3CosMul, PairDir)
"""
def evaluateAnalogy(embedding, test_file_name, p1, p2, q1, q2, all_score, all_combination):
    results = []

    # Similaridad 3CosAdd
    sim_cos_add = 0

    sim_cos_add += getCoseneSimilar(embedding, p1, p2, q1, q2)
    if all_combination:
        sim_cos_add += getCoseneSimilar(embedding, q1, q2, p1, p2)

        # Algunas relaciones pueden no ser biyectivas
        if not test_file_name in RESTRICTED_FILES:
            sim_cos_add += getCoseneSimilar(embedding, p2, p1, q2, q1)
            sim_cos_add += getCoseneSimilar(embedding, q2, q1, p2, p1)

    results.append(sim_cos_add)

    if not all_score:
        results = results + (6*[0])
        return results


    # Similaridad 3CosMul
    sim_cos_mul = 0

    sim_cos_mul += getCoseneSimilarCosmul(embedding, p1, p2, q1, q2)
    if all_combination:
        sim_cos_mul += getCoseneSimilarCosmul(embedding, q1, q2, p1, p2)

        # Algunas relaciones pueden no ser biyectivas
        if not test_file_name in RESTRICTED_FILES:
            sim_cos_mul += getCoseneSimilarCosmul(embedding, p2, p1, q2, q1)
            sim_cos_mul += getCoseneSimilarCosmul(embedding, q2, q1, p2, p1)

    results.append(sim_cos_mul)


    # TODO: Similaridad PairDir
    results.append(0)


    # Puntaje coseno
    results.append(getCos(embedding, p1, p2, q1, q2))

    # Puntaje euclidiano
    results.append(getEuc(embedding, p1, p2, q1, q2))

    # Puntaje n-coseno
    results.append(getNCos(embedding, p1, p2, q1, q2))

    # Puntaje n-euclidiano
    results.append(getNEuc(embedding, p1, p2, q1, q2))

    return results


#
def analogyTest(embedding, embedding_name, all_score=False, all_combination=False):
    # Obtencion de path a test
    test_file_list = getTestFiles()
    test_file_list = getUntestedFiles(test_file_list, embedding_name)

    for file in test_file_list:
        print(file.name)

    # Revisamos todos los archivos para realizar test
    for file in test_file_list:
        print("Testing: ", end='')
        print(file.name)
        pair_list = getAnalogyPairs(file)

        #for pair in pair_list:
        #    print(pair)

        # Inicializamos variables para guardar metricas, calcular accuracy de similaridad y puntaje definido por Che
        total_test_result = []
        similarity_total = [0, 0, 0]
        che_metric = [0, 0, 0, 0]

        count_relations = 0
        count_multiply = 1
        if all_combination:
            if not file.name in RESTRICTED_FILES:
                count_multiply = 4
            else:
                count_multiply = 2

        # Generamos todas las 4-tuplas posibles a partir de todos los pares presentes en el archivo file
        for i in range(len(pair_list)):
            for j in range(len(pair_list)):
                if i == j:
                    continue

                # Contar la cantidad de relaciones que se pueden hacer con todas las tuplas posibles
                count_relations += 1

                # Generamos las tuplas p1:p2 como q1:q2
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
                result_tuple = evaluateAnalogy(embedding, file.name, p1, p2, q1, q2, all_score, all_combination)

                # Separamos los resultados:
                # Similaridad
                similarity_total[0] += result_tuple[0]
                similarity_total[1] += result_tuple[1]
                similarity_total[2] += result_tuple[2]

                # Puntajes definido por Che
                che_metric[0] += result_tuple[3]
                che_metric[1] += result_tuple[4]
                che_metric[2] += result_tuple[5]
                che_metric[3] += result_tuple[6]

        # Calculamos los resultados totales del test
        # Similaridad
        total_test_result.append(["3CosAdd", similarity_total[0] * 1.0 / (1.0 * count_relations * count_multiply)])
        total_test_result.append(["3CosMul", similarity_total[1] * 1.0 / (1.0 * count_relations * count_multiply)])
        total_test_result.append(["PairDir", similarity_total[2] * 1.0 / (1.0 * count_relations * count_multiply)])

        # Puntajes definido por Che
        total_test_result.append(["cos", che_metric[0] * 1.0 / (1.0 * count_relations)])
        total_test_result.append(["euc", che_metric[1] * 1.0 / (1.0 * count_relations)])
        total_test_result.append(["ncos", che_metric[2] * 1.0 / (1.0 * count_relations)])
        total_test_result.append(["neuc", che_metric[3] * 1.0 / (1.0 * count_relations)])

        # Guardamos los resultados de forma temporal
        saveTempResults(embedding_name, file.name, total_test_result)

    saveResults(embedding_name)

