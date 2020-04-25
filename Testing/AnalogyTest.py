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

DEBUG = True

# Retorna 1 o 0, dependiendo de si se encontro determino la analogia
def get_cosene_similar_cosmul(embedding, p1, p2, q1, q2):
    for a in p1:
        for b in p2:
            for c in q1:
                res = 0
                try:
                    res = embedding.most_similar_cosmul(positive=[b, c], negative=[a])
                except:
                    res = [['']]

                res = res[0][0]
                if DEBUG:
                    print(">>> " + res)

                if res in q2:
                    return 1

    return 0

def get_cosene_similar(embedding, p1, p2, q1, q2):
    for a in p1:
        for b in p2:
            for c in q1:
                res = 0
                try:
                    res = embedding.most_similar(positive=[b, c], negative=[a])
                except:
                    res = [['']]

                res = res[0][0]
                if DEBUG:
                    print(">>> " + res)

                if res in q2:
                    return 1

    return 0

def get_cos(embedding, p1, p2, q1, q2):
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


def get_euc(embedding, p1,p2, q1, q2):
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


def get_n_cos(embedding, a, b, c, d):
    pass


def get_n_euc(embedding, a, b, c, d):
    pass


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
    # Calculo de similaridad 3CosMul
    res_sim_cosmul = get_cosene_similar_cosmul(embedding, p1, p2, q1, q2)
    cant_sim_cosmul = 1

    if all_variation:
        res_sim_cosmul += get_cosene_similar_cosmul(embedding, q1, q2, p1, p2)
        cant_sim_cosmul += 1

        if file_name not in RESTRICTED_RELATIONS:
            res_sim_cosmul += get_cosene_similar_cosmul(embedding, q1, q2, p1, p2)
            res_sim_cosmul += get_cosene_similar_cosmul(embedding, q1, q2, p1, p2)

            cant_sim_cosmul += 2



    results = [[res_sim_cosmul, cant_sim_cosmul]]

    if not all_scores:
        return results

    #TODO: calculo de otras metricas


    return results



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

    count = 0

    print("all_variation: ", end='')
    print(all_variation)
    print("all_scores: ", end='')
    print(all_scores)

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

        # Realizamos las analogias para cada par de relaciones.
        results = [[0, 0]]
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue

                if i == 0 and DEBUG:
                    print(test_pairs[i], end=' ')
                    print(test_pairs[j])

                p = test_pairs[i]
                p1 = p[0].split('/')
                p2 = p[1].split('/')

                q = test_pairs[j]
                q1 = q[0].split('/')
                q2 = q[1].split('/')

                # Evaluamos la 4-tupla.
                res = evaluation(p1, p2, q1, q2, embedding, test_file.split("\\")[-1], all_variation, all_scores)

                if i == 0 and DEBUG:
                    print(res)

                # Resultados de similaridad 3CosMul
                results[0][0] += res[0][0]
                results[0][1] += res[0][1]

                # Resultados de similaridad 3CosAdd


        print(results[0][0], end='/')
        print(results[0][1])
        print("accuracy: ", end='')
        print(results[0][0] / results[0][1])

        break

    return results

