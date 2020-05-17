import io
import os
import random

import Constant
import networkx


_RESULT = Constant.RESULTS_FOLDER / "CrossMatch"

import random
import math
import networkx as nx


###########################################################################################
# PUNTUACION Y MATCHING BIPARTITO
###########################################################################################


"""
A partir de una matriz de distancias, este metodo define un grafo

:para matrix: matriz de distancias

:return: grafo bidireccional con pesos, correspondiente a la matriz dada
"""
def getGraph(matrix):
    G = nx.Graph()

    for i in range(len(matrix)):
        G.add_node(i)
        for j in range(len(matrix)):
            if j <= i:
                continue

            G.add_edge(i, j)
            G[i][j]["weight"] = matrix[i][j]
            G[j][i]["weight"] = matrix[j][i]

    print("Graph description: ")
    print(G.nodes)
    print(G.edges)

    return G


"""
Calcula la distancia que existe entre dos palabras, pertenecientes a embedding distintos

:param embedding1: lista de vectores de palabras
:param embedding2: lista de vectores de palabras
:param word1: palabra dentro de embedding1
:param word2: palabra dentro de embedding2

:return: valor de distancia entre dos vectores
"""
def distance(embedding1, embedding2, word1, word2):
    w_vector1 = embedding1.word_vec(word1, True)
    w_vector2 = embedding2.word_vec(word2, True)

    return w_vector1.dot(w_vector2) - 1


"""
Obtencion de sampling de un embedding

:param embedding: lista de vectores de palabras
:param sample_size: tamaño del sampling principal

:return: lista de palabras existentes embedding 
"""
def getTestSet(embedding, sample_size):
    word_list = []

    for w, obj in embedding.vocab.items():
        word_list.append(w)

    sample = random.sample(word_list, sample_size)

    return sample


"""
Obtencion de constante usada durante metrica, este metodo fue hecho para evitar multiples calculos.

:param sub_sample_size: tamaño del sampling a utilizar en las mediciones

:return: constante utilizada durante el calculo de metricas
"""
def getFConstant(sub_sample_size):
    F_constant = (math.factorial(sub_sample_size)**3) / (math.factorial(2 * sub_sample_size))

    return F_constant


"""
Calculo de metrica

:param M: matching bipartito asociado al grafo de distancia entre sampling de los distintos embeddings
:param sub_sample_size: tamaño del sampling a utilizar en las mediciones
:param F_constant: constante que se utiliza durante el calculo de la metrica, se obtiene previamente debido a la naturaleza de la metrica.

:return: puntaje asociado a la metrica de distancia
"""
def getScore(M, sub_sample_size, F_constant):
    c1 = 0
    for pair in M:
        if pair[0] < sub_sample_size and pair[1] >= sub_sample_size:
            c1 += 1

        if pair[1] < sub_sample_size and pair[0] >= sub_sample_size:
            c1 += 1

    print("#Cross pair: " + str(c1))
    sum = 0
    for c in range(c1+1):
        if (sub_sample_size - c) % 2 == 1:
            continue

        c0 = (sub_sample_size - c) / 2
        c2 = (sub_sample_size - c) / 2

        sum += ((2**c) / (math.factorial(c0) * math.factorial(c) * math.factorial(c2)))

    return (sum * F_constant)


"""
Obtencion de matriz de distancia asociado al subsampling

:param embedding1: lista de vectores de palabras
:param embedding2: lista de vectores de palabras
:param sample1: sampling de palabras asociado a embedding1
:param sample2: sampling de palabras asociado a embedding1
:param sub_sample_size: tamaño de subsampling

:return: matriz de distancia
"""
def getDistanceMatrix(embedding1, embedding2, sample1, sample2, sub_sample_size):
    sub_sample1 = random.sample(sample1, sub_sample_size)
    print("sub sample 1: ", end='')
    print(sub_sample1)

    sub_sample2 = random.sample(sample2, sub_sample_size)
    print("sub sample 2: ", end='')
    print(sub_sample2)

    sample = sub_sample1 + sub_sample2

    distance_matrix = [[10000000 for x in range(2*sub_sample_size)] for y in range(2*sub_sample_size)]

    for i in range(len(sample)):
        for j in range(len(sample)):
            if j <= i:
                continue

            if i < len(sub_sample1):
                emb1 = embedding1
            else:
                emb1 = embedding2

            if j < len(sub_sample1):
                emb2 = embedding1
            else:
                emb2 = embedding2

            distance_matrix[i][j] = distance(emb1, emb2, sample[i], sample[j])
            distance_matrix[j][i] = distance_matrix[i][j]

    return distance_matrix


###########################################################################################
# GUARDAR RESULTADOS
###########################################################################################


"""
Metodo para guardar resultados

:param embedding1_name: nombre de uno de los embeddings
:param embedding2_name: nombre de uno de los embeddings
:param results: resultados asociados a los dos embeddings testeados
"""
def saveResult(embedding1_name, embedding2_name, results):
    if not _RESULT.exists():
        os.makedirs(_RESULT)

    result_file = _RESULT / (embedding1_name + "_" + embedding2_name + ".txt")
    with io.open(result_file, 'w', encoding='utf-8') as f:
        f.write(str(results) + "\n")


###########################################################################################
# EVALUACION POR OUTLIER DETECTION
###########################################################################################


"""
Realizacion test de cross-matching

:param embedding1: lista de vectores de palabras
:param embedding1_name: nombre asociado a embedding1
:param embedding2: lista de vectores de palabras
:param embedding2_name: nombre asociado a embedding2
:param sample_size: tamaño de sampling principal
:param sub_sample_size: tamaño de sub-sampling
:param repetitions: cantidad de repeticion del test
:param F_constant: constante asociada a la metrica utilizada

:return: resultados de test de cross-matching asociado a los embeddings
"""
def crossMatchTest(embedding1, embedding1_name, embedding2, embedding2_name, sample_size=100000, sub_sample_size=200, repetitions=500, F_constant=None):
    sample1 = getTestSet(embedding1, sample_size)
    sample2 = getTestSet(embedding2, sample_size)

    sum = 0
    if F_constant == None:
        F_constant = getFConstant(sub_sample_size)

    for i in range(repetitions):
        matrix = getDistanceMatrix(embedding1, embedding2, sample1, sample2, sub_sample_size)

        G = getGraph(matrix)
        M = nx.max_weight_matching(G, True)

        #print("matching pairs:")
        #for pair in M:
        #    print("> " + str(pair[0]) + " - " + str(pair[1]))

        p_value = getScore(M, sub_sample_size, F_constant)
        print("p-value " + str(i) + ": " + str(p_value))

        sum += p_value

    result = (sum * 1.0 / repetitions)
    saveResult(embedding1_name, embedding2_name, result)

    print("result: " + str(result))

    return result