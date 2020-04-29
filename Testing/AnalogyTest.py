import os
import io
import shutil

import numpy as np

from pathlib import Path

CURRENT_PATH = os.getcwd()
path = Path(CURRENT_PATH)

###########################################################################################
# METRICAS
###########################################################################################

"""
Retorna 1 o 0 si se logra determinar la palabra d, dentro de relacion a:b = c:d
utilizando 3CosMul como funcion de similaridad
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
"""
def getCos(embedding, p1, p2, q1, q2):
    result = -1.0
    for a in p1:
        for b in p2:
            for c in q1:
                for d in q2:
                    # TODO: caso en que palabra no esta en embedding
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
"""
def getEuc(embedding, p1,p2, q1, q2):
    result = -1.0
    for a in p1:
        for b in p2:
            for c in q1:
                for d in q2:
                    # TODO: caso en que palabra no esta en embedding
                    a_vec = embedding[a]
                    b_vec = embedding[b]
                    c_vec = embedding[c]
                    d_vec = embedding[d]

                    r = np.linalg((b_vec - a_vec) - (d_vec - c_vec)) / (np.linalg.norm(b_vec - a_vec) + np.linalg.norm(d_vec - c_vec))
                    if r > result:
                        result = r

    return result

"""
Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia coseno como funcion de puntaje, en este
caso, los vectores son unitarios
"""
def getNCos(embedding, p1,p2, q1, q2):
    pass

"""
Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia euclidiana como funcion de puntaje, en este
caso, los vectores son unitarios
"""
def getNEuc(embedding, p1,p2, q1, q2):
    pass


def getPairDist(embedding, p1,p2, q1, q2):
    pass


###########################################################################################
# FILES HANDLING
###########################################################################################

"""
Obtencion de path hacia los distintos archivos de test de analogias
"""
def getTestFiles():
    dataset_folder = path.parent / "Datasets/AnalogyDataset"

    if not dataset_folder.exists():
        raise Exception('Dataset folder not found')

    test_files = []
    sub_dataset_folder = os.listdir(dataset_folder)
    for folder in sub_dataset_folder:
        sub_folder = dataset_folder / folder

        test_files = test_files + list(map(lambda f: sub_folder / f, os.listdir(sub_folder)))

    return test_files


"""
Obtencion de path hacia los dintintos archivos de test de analogias que no hayan sido evaluados aun
"""
def getUntestedFiles(test_files, embedding_name):
    temp_result_path = path.parent / "TempResults/Analogy"

    #if not temp_result_path.exists():
    #    return test_files

    test_files_list = []
    for file in test_files:
        #print(file)
        temp_result_file_path = temp_result_path / embedding_name / file.name
        #print(temp_result_file_path)

        if not temp_result_file_path.exists():
            test_files_list.append(file)

    return test_files_list


###########################################################################################
# SAVE RESULTS
###########################################################################################

"""
Guarda resultados de analogias de forma temporal
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
"""
def saveResults(embedding_name):
    temp_analogy_results_folder = path.parent / "TempResults/Analogy"
    temp_result_embedding = temp_analogy_results_folder / embedding_name

    if not temp_result_embedding.exists():
        return

    test_result_list = os.listdir(temp_result_embedding)
    results = []
    for test_file_name in test_result_list:
        test_result_file = temp_result_embedding / test_file_name

        aux_result = []
        with io.open(test_result_file, 'r') as f:
            for line in f:
                aux_result.append(line.strip().split())

        results.append([test_file_name, aux_result])

    print(results)

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

#
def evaluateAnalogy():
    pass


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


#
def analogyTest(embedding, embedding_name):
    # Obtencion de path a test
    test_file_list = getTestFiles()
    test_file_list = getUntestedFiles(test_file_list, embedding_name)

    for file in test_file_list:
        print(file)

    for file in test_file_list:
        pair_list = getAnalogyPairs(file)
        for pair in pair_list:
            print(pair)

        break
    pass


analogyTest("w2", "asdf")