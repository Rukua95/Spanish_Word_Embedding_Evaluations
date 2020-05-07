import io
import os
import random

import Constant
import networkx


_RESULT = Constant.RESULTS_FOLDER / "CrossMatch"

import random
import networkx as nx

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

    print(G.nodes)
    print(G.edges)

    return G

def distance(embedding1, embedding2, word1, word2):
    w_vector1 = embedding1.word_vec(word1, True)
    w_vector2 = embedding2.word_vec(word2, True)

    return w_vector1.dot(w_vector2) - 1


def getTestSet(embedding, sample_size):
    word_list = []

    for w, obj in embedding.vocab.items():
        word_list.append(w)

    sample = random.sample(word_list, sample_size)

    return sample


def getScore(M, sub_sample_size):
    return 0


def getDistanceMatrix(embedding1, embedding2, sample1, sample2, sub_sample_size):
    sub_sample1 = random.sample(sample1, sub_sample_size)
    print(sub_sample1)
    sub_sample2 = random.sample(sample2, sub_sample_size)
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
            #print(distance_matrix[i][j], end=' ')

        #print('\n')

    return distance_matrix

def saveResult(embedding1_name, embedding2_name, results):
    if not _RESULT.exists():
        os.makedirs(_RESULT)

    result_file = _RESULT / (embedding1_name + "_" + embedding2_name + ".txt")
    with io.open(result_file, 'w', encoding='utf-8') as f:
        f.write(str(results) + "\n")


def crossMatchTest(embedding1, embedding1_name, embedding2, embedding2_name, sample_size=100000, sub_sample_size=200, repetitions=500):
    sample1 = getTestSet(embedding1, sample_size)
    sample2 = getTestSet(embedding2, sample_size)

    sum = 0

    for i in range(repetitions):
        matrix = getDistanceMatrix(embedding1, embedding2, sample1, sample2, sub_sample_size)

        for l in matrix:
            print(l)

        G = getGraph(matrix)
        M = nx.max_weight_matching(G, True)
        for pair in M:
            print(str(pair[0]) + " - " + str(pair[1]))

        p_value = getScore(M, sub_sample_size)
        print("p-value " + str(i) + ": " + str(p_value))

        sum += p_value

    result = (sum * 1.0 / repetitions)

    return result