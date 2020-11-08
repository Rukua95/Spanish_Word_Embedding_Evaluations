from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import networkx as nx

import random
import math
import os
import io
import Constant


# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

class CrossMatchTestClass:

    # Resultados
    _RESULT = Constant.RESULTS_FOLDER / "CrossMatch"

    def __init__(self, cantidad=None, main_sample=100000, sub_sample_size=200, repetitions=500):
        print("Test de CrossMatch")

        self._embeddings_size = cantidad
        self._main_sample = main_sample
        self._sub_sample_size = sub_sample_size
        self._repetitions = repetitions

        self._F_constant = self.getFConstant(sub_sample_size)


    ###########################################################################################
    # PUNTUACION Y MATCHING BIPARTITO
    ###########################################################################################


    """
    A partir de una matriz de distancias, este metodo define un grafo
    
    :para matrix: matriz de distancias
    
    :return: grafo bidireccional con pesos, correspondiente a la matriz dada
    """
    def getGraph(self, matrix):
        G = nx.Graph()

        for i in range(len(matrix)):
            G.add_node(i)
            for j in range(len(matrix)):
                if j <= i:
                    continue

                G.add_edge(i, j)
                G[i][j]["weight"] = matrix[i][j]
                G[j][i]["weight"] = matrix[j][i]

        #print("Graph description: ")
        #print(G.nodes)
        #print(G.edges)

        return G


    """
    Calcula la distancia que existe entre dos palabras, pertenecientes a embedding distintos
    
    :param embedding1: lista de vectores de palabras
    :param embedding2: lista de vectores de palabras
    :param word1: palabra dentro de embedding1
    :param word2: palabra dentro de embedding2
    
    :return: valor de distancia entre dos vectores
    """
    def distance(self, embedding1, embedding2, word1, word2):
        x = embedding1[word1]
        y = embedding2[word2]

        return 1 + np.dot(x,y) / (np.sqrt(np.dot(x,x)) * np.sqrt(np.dot(y,y)))


    """
    Obtencion de sampling de un embedding
    
    :param embedding: lista de vectores de palabras
    :param sample_size: tamaño del sampling principal
    
    :return: lista de palabras existentes embedding 
    """
    def getWordList(self, embedding):
        #word_list = []
        #for w, obj in embedding.vocab.items():
        #    word_list.append(w)

        word_list = embedding.getWordList()

        sample = random.sample(word_list, self._main_sample)

        return sample


    """
    Obtencion de constante usada durante metrica, este metodo fue hecho para evitar multiples calculos.
    
    :param sub_sample_size: tamaño del sampling a utilizar en las mediciones
    
    :return: constante utilizada durante el calculo de metricas
    """
    def getFConstant(self, sub_sample_size):
        F_constant = (math.factorial(sub_sample_size)**3) / (math.factorial(2 * sub_sample_size))

        return F_constant


    """
    Calculo de metrica
    
    :param M: matching bipartito asociado al grafo de distancia entre sampling de los distintos embeddings
    :param sub_sample_size: tamaño del sampling a utilizar en las mediciones
    :param F_constant: constante que se utiliza durante el calculo de la metrica, se obtiene previamente debido a la naturaleza de la metrica.
    
    :return: puntaje asociado a la metrica de distancia
    """
    def getScore(self, M):
        c1 = 0
        for pair in M:
            if pair[0] < self._sub_sample_size and pair[1] >= self._sub_sample_size:
                c1 += 1

            if pair[1] < self._sub_sample_size and pair[0] >= self._sub_sample_size:
                c1 += 1

        #print("#Cross pair: " + str(c1))
        sum = 0
        for c in range(c1+1):
            if (self._sub_sample_size - c) % 2 == 1:
                continue

            c0 = (self._sub_sample_size - c) / 2
            c2 = (self._sub_sample_size - c) / 2

            sum += ((2**c) / (math.factorial(c0) * math.factorial(c) * math.factorial(c2)))

        return (sum * self._F_constant), c1


    """
    Obtencion de matriz de distancia asociado al subsampling
    
    :param embedding1: lista de vectores de palabras
    :param embedding2: lista de vectores de palabras
    :param sample1: sampling de palabras asociado a embedding1
    :param sample2: sampling de palabras asociado a embedding1
    :param sub_sample_size: tamaño de subsampling
    
    :return: matriz de distancia
    """
    def getDistanceMatrix(self, embedding1, embedding2, sample1, sample2):
        sub_sample1 = random.sample(sample1, self._sub_sample_size)
        sub_sample2 = random.sample(sample2, self._sub_sample_size)

        sample = sub_sample1 + sub_sample2

        distance_matrix = [[10000000 for x in range(2*self._sub_sample_size)] for y in range(2*self._sub_sample_size)]

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

                distance_matrix[i][j] = self.distance(emb1, emb2, sample[i], sample[j])
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
    def saveResult(self, embedding1_name, embedding2_name, results):
        if not self._RESULT.exists():
            os.makedirs(self._RESULT)

        result_file = self._RESULT / (embedding1_name + "_" + embedding2_name + ".txt")
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
    def crossMatchTest(self, word_embedding1, word_embedding_name1, word_embedding2, word_embedding_name2):
        results = {}

        word_list1 = self.getWordList(word_embedding1)
        word_list2 = self.getWordList(word_embedding2)

        sum = 0

        print("Testing " + word_embedding_name1 + " y " + word_embedding_name2)
        for h in range(self._repetitions):
            matrix = self.getDistanceMatrix(word_embedding1, word_embedding2, word_list1, word_list2)

            G = self.getGraph(matrix)
            M = nx.max_weight_matching(G, True)

            p_value, c1 = self.getScore(M)
            if (h+1) % 10 == 0:
                print("p-value " + str(h) + ": " + str(p_value))

            sum += p_value

        pair_result = sum / self._repetitions
        results[word_embedding_name1 + "__" + word_embedding_name2] = pair_result
        self.saveResult(word_embedding_name1, word_embedding_name2, pair_result)

        print("Result: " + str(pair_result), end='\n\n')

        return results