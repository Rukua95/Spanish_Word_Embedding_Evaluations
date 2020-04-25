import os
import io

import numpy as np

RES_PATH = "D:\\Documents\\Memoria - Eval. Word Embeddings\\Data\\Results_Bats"
PATH = "D:\\Documents\\Memoria - Eval. Word Embeddings\\Data\\BATS_español"
DERIVACION = "Derivacion"
INFLEXION = "Enciclopedia"
ENCICLOPEDIA = "Inflexion"
LEXICOGRAFICO = "Lexicografico"

DATA_TYPE = [DERIVACION, INFLEXION, ENCICLOPEDIA, LEXICOGRAFICO]
RESTRICTED_RELATIONS = []

# Most similar vector using cosene similarity
def get_cosene_similar_cosmul(embedding, a, b, c):
    res = embedding.most_similar_cosmul(positive=[b, c], negative=[a])
    return res[0][0]

def get_cosene_similar(embedding, a, b, c):
    res = embedding.most_similar(positive=[b, c], negative=[a])
    return res[0][0]

def get_cos(embedding, a, b, c, d):
    a = embedding[a]
    b = embedding[b]
    c = embedding[c]
    d = embedding[d]

    res = np.dot(b - a, d - c) / (np.linalg.norm(b - a)*np.linalg.norm(d - c))
    return res


def get_euc(embedding, a, b, c, d):
    a = embedding[a]
    b = embedding[b]
    c = embedding[c]
    d = embedding[d]

    res = np.linalg((b - a) - (d - c)) / (np.linalg.norm(b - a) + np.linalg.norm(d - c))
    return res


def get_n_cos(embedding, a, b, c, d):
    a = embedding.word_vec(a, True)
    b = embedding.word_vec(b, True)
    c = embedding.word_vec(c, True)
    d = embedding.word_vec(d, True)

    res = np.dot(b - a, d - c) / (np.linalg.norm(b - a) * np.linalg.norm(d - c))
    return res


def get_n_euc(embedding, a, b, c, d):
    a = embedding.word_vec(a, True)
    b = embedding.word_vec(b, True)
    c = embedding.word_vec(c, True)
    d = embedding.word_vec(d, True)

    res = np.linalg((b - a) - (d - c)) / (np.linalg.norm(b - a) + np.linalg.norm(d - c))
    return res


def pair_dist(embedding, a, b, c, d):
    pass


######################################################################################################

# Obtencion de path hacia los distintos archivos de test de analogias
def get_test_files():
    test_files = []

    for type in DATA_TYPE:
        test_path = PATH + "\\" + type
        print(test_path)

        for file_name in os.listdir(test_path):
            test_files.append(test_path + "\\" + file_name)



    return test_files


# Retorna los resultados de cada metrica para la tupla dada
def evaluation(p1, p2, q1, q2, embedding, file_name, all_variation, all_scores):
    puntajes_Che = [-1, -1, -1, -1]
    puntaje_sim_cos = [0, 0]
    cantidad_relaciones = [1, 1]
    if all_variation:
        if file_name in RESTRICTED_RELATIONS:
            cantidad_relaciones = [2, 2]
        else:
            cantidad_relaciones = [4, 4]

    verificar_cos_add = False
    verificar_cos_mul = False
    for a in p1:
        for b in p2:
            for c in q1:
                r = get_cosene_similar_cosmul(embedding, a, b, c)
                if r in q2 and not verificar_cos_add:
                    puntaje_sim_cos[0] += 1
                    verificar_cos_add = True

                if not all_scores:
                    continue

                r = get_cosene_similar(embedding, a, b, c)
                if r in q2 and not verificar_cos_mul:
                    puntaje_sim_cos[0] += 1
                    verificar_cos_mul = True

                for d in q2:
                    # TODO: elegir el maximo
                    puntajes_Che[0] = get_cos(embedding, a, b, c, d)
                    puntajes_Che[1] = get_euc(embedding, a, b, c, d)
                    puntajes_Che[2] = get_n_cos(embedding, a, b, c, d)
                    puntajes_Che[3] = get_n_euc(embedding, a, b, c, d)

    if not all_variation:
        return

    # TODO: hacer todas las relacions


# Analogy test
def analogy_test(embedding, name, all_variation=False, all_scores=False):
    # En caso de que existan fallos durante el calculo de los puntajes, los puntajes se guardan en archivos separados
    # y se continua en el siguiente despues del ultimo guardado.
    test_files = get_test_files()

    result_path = RES_PATH
    last_results = os.listdir(result_path)
    aux = []
    for file in last_results:
        if name in file:
            aux.append(file)

    last_results = aux
    aux = []
    print("Results saved:")
    print(last_results)
    for file in test_files:
        ver = True
        for res_name in last_results:
            if file.split("\\")[-1] in res_name:
                ver = False

        if ver:
            aux.append(file)

    # Archivos con los test a usar para embedding.
    test_files = aux
    print("Test files: ")
    print(test_files)

    results = {}

    count = 1

    # Realizacion de los test
    for file in test_files:

        # Extraccion de los pares de palabras.
        test_file = file
        count += 1
        test_pairs = []
        print("Testing file number " + str(count) + " of " + str(len(test_files)) + ": " + test_file.split("\\")[-1])

        with io.open(test_file, 'r') as f:
            for line in f:
                pair = line.strip().split()
                test_pairs.append(pair)

        n = len(test_pairs)

        answer = []
        solution = []

        # Realizamos las analogias para cada par de relaciones.
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue

                p = test_pairs[i]
                p1 = p[0].split('/')
                p2 = p[1].split('/')

                q = test_pairs[j]
                q1 = q[0].split('/')
                q2 = q[1].split('/')

                q2 = q[1]

                # Evaluamos la 4-tupla.
                result = evaluation(p1, p2, q1, q2, embedding, test_file.split("\\")[-1], all_variation, all_scores)

                # result es un diccionario con los distintos resultados.
                # -> puntaje de mas similar: accuracy / cantidad de tuplas
                # -> puntaje cos, euc, n_cos, n_euc: suma de puntajes / cantidad de tuplas

                # TODO: procesar el puntaje parcial

        # TODO: procesar el puntaje total

        #with io.open(result_path + "\\" + name + "_" + test_file.split("\\")[-1], 'w') as f:
        #    f.write(str(accuracy))

        #print(name + "_" + test_file.split("\\")[-1])
        #print("...Complete with score: " + str(accuracy))



    return results

