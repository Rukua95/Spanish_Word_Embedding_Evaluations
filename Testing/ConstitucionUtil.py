import csv
import re


import GPUtil

import torch
import torch.nn as nn

from random import shuffle

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from pytorchtools import EarlyStopping
from ConstitucionDataHandling import getSortedDataset


import ConstitucionUtil

from gensim.models.keyedvectors import KeyedVectors

import os
import io
import numpy as np

import Constant

# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file
    print(">>> Cargando vectores " + file + " ...", end='')
    word_vector = KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)
    print("listo.\n")

    return word_vector

_DATASET = Constant.DATA_FOLDER / "_Constitucion\\constitucion_data.csv"
_RESULT = Constant.RESULTS_FOLDER / "Constitucion"




###########################################################################################
# Clasificacion a partir de vectores promedio
###########################################################################################

class ConstitucionTestClass:
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _lower = True
    _oov_word = {}

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "_Constitucion"
    _RESULT = Constant.RESULTS_FOLDER / "Constitucion"
    def __init__(self):
        print("Constitucion test class")


    def prepareTaskA(self, gob_concept_vectors, gob_args_vectors):

        # Conceptos

        # Inicializar diccionario, segun topico, con lista de vectores
        gob_concept_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de conceptos
        gob_concept_concept_list_by_topics = {}

        for topic in gob_concept_vectors.keys():
            gob_concept_vectors_list_by_topics[topic] = []
            gob_concept_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto de gobierno.
        for topic in gob_concept_vectors.keys():
            print("Topico " + topic + ", cantidad de conceptos: " + str(len(gob_concept_vectors[topic].keys())))

            # Guardamos vectores y strings de conceptos
            for concept in gob_concept_vectors[topic].keys():
                if gob_concept_vectors[topic][concept].size == 0:
                    continue

                # Guardando vectores
                gob_concept_vectors_list_by_topics[topic].append(gob_concept_vectors[topic][concept])

                # Guardar concepts
                gob_concept_concept_list_by_topics[topic].append(concept)

            gob_concept_vectors_list_by_topics[topic] = np.vstack(gob_concept_vectors_list_by_topics[topic])


        # Argumentos
        gobc_arguments_vectors_list_by_topics = {}
        gobc_arguments_concept_list_by_topics = {}

        for topic in gob_args_vectors.keys():
            gobc_arguments_vectors_list_by_topics[topic] = []
            gobc_arguments_concept_list_by_topics[topic] = []

        for topic in gob_args_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(gob_args_vectors[topic])))

            for tupla in gob_args_vectors[topic]:
                concept = tupla["concept"]
                args_content = tupla["arg"]["content"]
                args_vector = tupla["arg"]["vector"]

                # Revisar que concepto abierto entregado no es nulo
                if args_content.lower() == 'null':
                    continue

                # Revisar que el concepto abierto tiene un vector promedio que lo represente
                if args_vector.size == 0:
                    continue

                gobc_arguments_vectors_list_by_topics[topic].append(args_vector)
                gobc_arguments_concept_list_by_topics[topic].append(concept)

        return [gobc_arguments_vectors_list_by_topics, gobc_arguments_concept_list_by_topics], [gob_concept_vectors_list_by_topics, gob_concept_concept_list_by_topics]


    def prepareTaskB(self, gob_concept_vectors, open_args_vectors):

        # Conceptos

        # Inicializar diccionario, segun topico, con lista de vectores
        gob_concept_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de conceptos
        gob_concept_concept_list_by_topics = {}

        for topic in gob_concept_vectors.keys():
            gob_concept_vectors_list_by_topics[topic] = []
            gob_concept_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto de gobierno.
        for topic in gob_concept_vectors.keys():
            print("Topico " + topic + ", cantidad de conceptos: " + str(len(gob_concept_vectors[topic].keys())))

            # Guardamos vectores y strings de conceptos
            for concept in gob_concept_vectors[topic].keys():
                if gob_concept_vectors[topic][concept].size == 0:
                    continue

                # Guardando vectores
                gob_concept_vectors_list_by_topics[topic].append(gob_concept_vectors[topic][concept])

                # Guardar concepts
                gob_concept_concept_list_by_topics[topic].append(concept)

            gob_concept_vectors_list_by_topics[topic] = np.vstack(gob_concept_vectors_list_by_topics[topic])


        # Conceptos abiertos
        open_arguments_vectors_list_by_topics = {}
        open_arguments_concept_list_by_topics = {}

        for topic in open_args_vectors.keys():
            open_arguments_vectors_list_by_topics[topic] = []
            open_arguments_concept_list_by_topics[topic] = []

        for topic in open_args_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(open_args_vectors[topic])))

            for tupla in open_args_vectors[topic]:
                equivalent_concept = tupla["concept"]
                open_concept = tupla["open_concept"]["content"]
                open_concept_vector = tupla["open_concept"]["vector"]

                # Revisar que concepto esta dentro de los conceptos de gobierno
                if not equivalent_concept in gob_concept_concept_list_by_topics[topic]:
                    continue

                # Revisar que concepto abierto entregado no es nulo
                if open_concept.lower() == 'null':
                    continue

                # Revisar que el concepto abierto tiene un vector promedio que lo represente
                if open_concept_vector.size == 0:
                    continue

                open_arguments_vectors_list_by_topics[topic].append(open_concept_vector)
                open_arguments_concept_list_by_topics[topic].append(equivalent_concept)

        return [open_arguments_vectors_list_by_topics, open_arguments_concept_list_by_topics], [gob_concept_vectors_list_by_topics, gob_concept_concept_list_by_topics]


    def prepareTaskC(self, gob_args_vectors, open_args_vectors, mode_vectors):

        # Manejo de modos de argumentacion #

        # Lista de vectores
        mode_vectors_list = []

        # Lista de modos
        mode_name_list = []

        for mode in mode_vectors.keys():
            mode_vectors_list.append(mode_vectors[mode])
            mode_name_list.append(mode)


        # Manejo de conceptos abierto #

        # Inicializar diccionario, segun topico, con lista de vectores
        arguments_vectors_list = []
        # Inicializar diccionario, segun topico, con lista de conceptos abiertos
        arguments_mode_list = []

        # Obtenemos los vectores correspondientes a cada concepto abierto.
        for topic in open_args_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(open_args_vectors[topic])))

            # Guardamos vectores y strings de conceptos abiertos
            for tupla in (open_args_vectors[topic] + gob_args_vectors[topic]):
                argument = tupla["arg"]["content"]
                arg_vector = tupla["arg"]["vector"]
                arg_mode = tupla["mode"]

                # Revisar que argumento es de un modo valido
                if arg_mode == "blank" or arg_mode == "undefined":
                    continue

                # Revisar que argumento entregado no es nulo
                if argument.lower() == 'null':
                    continue

                # Revisar que el argumento tiene un vector promedio que lo represente
                if arg_vector.size == 0:
                    continue

                # Guardando vectores
                arguments_vectors_list.append(arg_vector)

                # Guardando concepto equivalente
                arguments_mode_list.append(arg_mode)

        return [arguments_vectors_list, arguments_mode_list], [mode_vectors_list, mode_name_list]


    def saveResults(self, result_taskA, result_taskB, result_taskC, word_vector_name):
        save_path = self._RESULT

        if not save_path.exists():
            os.makedirs(save_path)

        result_path = save_path / ("mean_vector_" + word_vector_name + ".txt")

        print(">>> Guardando resultados en:\n     " + str(result_path))
        with io.open(result_path, 'w', encoding='utf-8') as f:
            f.write("Task A results\n")
            print(result_taskA)
            for key in result_taskA.keys():
                f.write("Topico " + key + "\n")
                tupla = result_taskA[key]

                f.write("Top1 " + str(tupla[0]) + " Top5 " + str(tupla[1]) + "\n")

            f.write("Task B results\n")
            for key in result_taskB.keys():
                f.write("Topico " + key + "\n")
                tupla = result_taskB[key]

                f.write("Top1 " + str(tupla[0]) + " Top5 " + str(tupla[1]) + "\n")

            f.write("Task C results\n")
            for key in result_taskC.keys():
                tupla = result_taskC[key]

                f.write("Presicion " + str(tupla[2]) + " Recall " + str(tupla[3]) + " F1 " + str(2 / (1/tupla[2] + 1/tupla[3])) + "\n")



    def MeanVectorEvaluation(self):
        print("\n>>> Inicio de test <<<\n")
        for embedding_name in self._embeddings_name_list:
            word_vector_name = embedding_name.split('.')[0]
            word_vector = get_wordvector(embedding_name, self._embeddings_size)

            # Obtencion de datos ordenados, ademas de sus respectivos vectores promedios.
            gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors = getSortedDataset(
                word_vector)

            print("Conceptos de gobierno por topico")
            for key in gob_concept_vectors.keys():
                print(key + ": " + str(len(gob_concept_vectors[key])))

            print("Argumentos para conceptos de gobierno")
            for key in gob_args_vectors.keys():
                print(key + ": " + str(len(gob_args_vectors[key])))

            print("Argumentos para conceptos abiertos")
            for key in open_args_vectors.keys():
                print(key + ": " + str(len(open_args_vectors[key])))

            print("Modos de argumentacion")
            for key in mode_vectors.keys():
                print(key + ": " + str(len(mode_vectors[key])))

            ######################################################################################
            # Task A

            print("\nTask A")
            gobc_arguments_vec_label, gob_concept_vectors_label = self.prepareTaskA(gob_concept_vectors, gob_args_vectors)
            result_taskA = self.meanVectorClasification(gobc_arguments_vec_label[0], gobc_arguments_vec_label[1], gob_concept_vectors_label[0], gob_concept_vectors_label[1])

            ######################################################################################
            # Task B

            open_concept_vector_label, gob_concept_vectors_label = self.prepareTaskB(gob_concept_vectors, open_args_vectors)
            print("\nTask B")
            result_taskB = self.meanVectorClasification(open_concept_vector_label[0], open_concept_vector_label[1], gob_concept_vectors_label[0], gob_concept_vectors_label[1])

            ######################################################################################
            # Task C
            arguments_vector_label, arg_mode_vectors_label = self.prepareTaskC(gob_args_vectors, open_args_vectors, mode_vectors)
            print("\nTask C")
            result_taskC = self.meanVectorClasification({"m": arguments_vector_label[0]}, {"m": arguments_vector_label[1]},
                                                        {"m": arg_mode_vectors_label[0]}, {"m": arg_mode_vectors_label[1]})

            # Guardamos resultados
            self.saveResults(result_taskA, result_taskB, result_taskC, word_vector_name)

        return result_taskA, result_taskB, result_taskC


    def meanVectorClasification(self, input_vectors, input_labels, class_vector, class_label):
        acuraccy_results = {}

        # Obtencion accuracy (top1 y top5) de similaridad.
        for topic in input_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(input_vectors[topic])))

            total_evaluado = len(input_vectors[topic])
            top5_correct = 0
            top1_correct = 0

            recall_values = {}
            presicion_values = {}

            for i in range(len(input_vectors[topic])):
                if (i + 1) % (len(input_vectors[topic]) // 10) == 0:
                    print(" > " + str(i) + ": top1_correct " + str(top1_correct) + ",top5_correct " + str(top5_correct))
                vector = input_vectors[topic][i]
                vector_label = input_labels[topic][i]

                # Comparando similaridad entre vectores promedios
                results = cosine_similarity(class_vector[topic], np.array([vector]))
                results = results.reshape(1, results.size)[0]

                index = np.argsort(results)
                index_most_similar = index[-1]
                index_most_similar_top5 = index[-5:]

                label1 = vector_label
                label2 = class_label[topic][index_most_similar]

                # Calculo de presicion y recall (pensado como resultado de task C)
                recall_values[label1] = [0, 0] if label1 not in recall_values.keys() else recall_values[label1]
                recall_values[label2] = [0, 0] if label2 not in recall_values.keys() else recall_values[label2]
                presicion_values[label1] = [0, 0] if label1 not in presicion_values.keys() else presicion_values[label1]
                presicion_values[label2] = [0, 0] if label2 not in presicion_values.keys() else presicion_values[label2]

                # Calcular si se predijo correctamente
                if label1 == label2:
                    top1_correct += 1

                    recall_values[label1][0] += 1
                    recall_values[label1][1] += 1
                    presicion_values[label1][0] += 1
                    presicion_values[label1][1] += 1

                else:
                    recall_values[label2][1] += 1
                    presicion_values[label1][1] += 1

                # Calcular si la prediccion es correcta en los primeros 5
                for id in index_most_similar_top5:
                    if vector_label == class_label[topic][id]:
                        top5_correct += 1
                        break


            # Calculo de accuracy para el topico
            top1_acuraccy = top1_correct / total_evaluado
            top5_acuraccy = top5_correct / total_evaluado

            presicion = 0
            recall = 0
            for key in presicion_values.keys():
                presicion += ((presicion_values[key][0] / presicion_values[key][1]) if presicion_values[key][1] != 0 else 0)
                recall += ((recall_values[key][0] / recall_values[key][1]) if recall_values[key][1] != 0 else 0)

            presicion = presicion / len(presicion_values.keys())
            recall = recall / len(presicion_values.keys())

            # Calculo de presicion y recall (Solo usado para task C)

            print("Resultados: " + str(top1_acuraccy) + " " + str(top5_acuraccy) + " " + str(presicion) + " " + str(recall))

            if topic not in acuraccy_results.keys():
                acuraccy_results[topic] = []

            acuraccy_results[topic] = [top1_acuraccy, top5_acuraccy, presicion, recall]

        return acuraccy_results

