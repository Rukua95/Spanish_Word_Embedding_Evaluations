import csv
import re

import io
import os
import shutil

import Constant

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import LongTensor
from torch.nn import Embedding, LSTM
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

import ConstitucionUtil

import random

_DATASET = Constant.DATA_FOLDER / "_Constitucion\\constitucion_data.csv"
_RESULT = Constant.RESULTS_FOLDER / "Constitucion"

def getDataset():
    data = []
    with io.open(_DATASET, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f, delimiter=',', escapechar='\\')
        header = csv_reader.fieldnames

        count = 0
        for line in csv_reader:
            data.append(line)
            count += 1

    return data, header


def getMeanVector(phrase, embedding, omit_oov=False):
    #print(phrase, end='\n > ')
    sum_vec = [np.zeros(embedding.vector_size)]
    phrase = re.sub('[^0-9a-zA-Záéíóú]+', ' ', phrase.lower())
    phrase = phrase.strip().split()
    num = len(phrase)
    #print(num)

    count = 0
    if num == 0:
        return np.array([]), count

    for word in phrase:
        if word not in embedding:
            count += 1
            if omit_oov:
                embedding.add(word, np.random.rand(embedding.vector_size))

            continue

        sum_vec.append(embedding[word])

    return [(np.sum(sum_vec, axis=0) / num), count]

"""
Entrega el dataset, ordenado para realizar evaluaciones segun vector promedio
"""
def getSortedDataset(embedding):
    data, header = getDataset()

    gob_concept_vectors = {}
    gob_args_vectors = {}
    open_args_vectors = {}
    mode_vectors = {}

    mode_vectors["policy"], _ = getMeanVector("político política", embedding)
    mode_vectors["value"], _ = getMeanVector("valor", embedding)
    mode_vectors["fact"], _ = getMeanVector("hecho factual", embedding)

    print("Size of dataset: " + str(len(data)))
    count = 0
    count_oov = 0
    for tuple in data:
        count += 1

        topic = tuple[header[0]]
        is_open_concept = tuple[header[1]]

        original_constitutional_concept = tuple[header[2]]
        original_constitutional_concept_vector, c = getMeanVector(original_constitutional_concept, embedding)
        count_oov += c

        constitutional_concept = tuple[header[3]]

        argument = tuple[header[4]]
        argument_vector, c = getMeanVector(argument, embedding)
        count_oov += c

        argument_mode = tuple[header[5]]

        if is_open_concept == 'no':
            if not topic in gob_concept_vectors.keys():
                gob_concept_vectors[topic] = {}
                gob_args_vectors[topic] = []

            if not constitutional_concept in gob_concept_vectors[topic].keys():
                gob_concept_vectors[topic][constitutional_concept], c = getMeanVector(constitutional_concept, embedding)
                count_oov += c

            gob_args_vectors[topic].append({
                "arg": {"content": argument, "vector": argument_vector},
                "concept": constitutional_concept,
                "mode": argument_mode,
            })


        else:
            if not topic in open_args_vectors.keys():
                open_args_vectors[topic] = []

            open_args_vectors[topic].append({
                "arg": {"content": argument, "vector": argument_vector},
                "concept": constitutional_concept,
                "open_concept": {"content": original_constitutional_concept, "vector": original_constitutional_concept_vector},
                "mode": argument_mode,
            })

        if count % 20000 == 0:
            print(str(count) + " " + argument)
            print("oov: " + str(count_oov))

    print(" > Dataset sorted")
    return gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors


###########################################################################################
# Clasificacion a partir de vectores promedio
###########################################################################################

class ConstitucionTestClass:
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

        result_path = save_path / (word_vector_name + ".txt")

        print(">>> Guardando resultados en:\n     " + str(result_path))
        with io.open(result_path, 'w', encoding='utf-8') as f:
            f.write("Task A results\n")
            for key in result_taskA.keys():
                f.write("Topico " + key + "\n")

                for tupla in result_taskA[key]:
                    f.write("Top1 " + tupla[0] + " Top5 " + tupla[1] + "\n")

            f.write("Task B results\n")
            for key in result_taskB.keys():
                f.write("Topico " + key + "\n")

                for tupla in result_taskA[key]:
                    f.write("Top1 " + tupla[0] + " Top5 " + tupla[1] + "\n")

            f.write("Task C results\n")
            for key in result_taskC.keys():

                for tupla in result_taskA[key]:
                    f.write("P " + tupla[0] + " R " + tupla[1] + " F1 " + tupla[2] + "\n")



    def MeanVectorEvaluation(self, word_vector, word_vector_name):
        # Obtencion de datos ordenados, ademas de sus respectivos vectores promedios.
        gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors = ConstitucionUtil.getSortedDataset(
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
        self.saveResults(result_taskA, result_taskA, result_taskA, word_vector_name)

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
                    print(" > " + str(i) + ": top1_correct = " + str(top1_correct) + " top5_correct" + str(top5_correct))
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

                presicion_values[label1] = [0, 0] if label1 not in presicion_values.keys() else
                presicion_values[label2] = [0, 0] if label2 not in presicion_values.keys() else
                recall_values[label1] = [0, 0] if label1 not in recall_values.keys() else
                recall_values[label2] = [0, 0] if label2 not in recall_values.keys() else

                # Calcular si se predijo correctamente
                if label1 == label2:
                    top1_correct += 1

                # Calcular si la prediccion es correcta en los primeros 5
                for id in index_most_similar_top5:
                    if vector_label == class_label[topic][id]:
                        top5_correct += 1
                        break


            # Calculo de accuracy para el topico
            top1_acuraccy = top1_correct / total_evaluado
            top5_acuraccy = top5_correct / total_evaluado

            # Calculo de presicion y recall (Solo usado para task C)

            print("Resultados: " + str(top1_acuraccy) + " " + str(top5_acuraccy))

            if topic not in acuraccy_results.keys():
                acuraccy_results[topic] = []

            acuraccy_results[topic] = [top1_acuraccy, top5_acuraccy]

        return acuraccy_results


#################################################################################
# Validacion con red neuronal
#################################################################################


class ClassifierModel(nn.Module):
    def __init__(self, label_size, emb_weight, traing_emb=False):
        super(ClassifierModel, self).__init__()

        if not traing_emb:
            with torch.no_grad():
                self.embedding = nn.Embedding(emb_weight.size()[0], emb_weight.size()[1])

        else:
            self.embedding = nn.Embedding(emb_weight.size()[0], emb_weight.size()[1])

        print(self.embedding)

        self.lstm = nn.LSTM(
            input_size=emb_weight.size()[1],
            hidden_size=label_size,
            bidirectional=False,
            batch_first=True)
        print(self.lstm)

        self.logsoftmax = nn.LogSoftmax(dim=1)
        print(self.logsoftmax)
        # self.softmax = nn.Softmax(dim=1)
        # print(self.softmax)

    def forward(self, entity_ids, seq_len):
        print(" >> forward")

        max_seq_len = torch.max(seq_len)
        print(" > max_len")
        print(max_seq_len)

        emb = self.embedding(entity_ids).cuda()
        print(" > emb")
        print(emb.size())

        out, _ = self.lstm(emb)
        out = out.cuda()
        print(" > out(max_len_seq x batch_len x output_size)")
        print(out.size())

        out = out.reshape(out.size()[0] * out.size()[1], out.size()[2]).cuda()
        print(" > out(reshape)")
        print(out.size())

        adjusted_lengths = [i * max_seq_len + l for i, l in enumerate(seq_len)]
        outputs_last = out.index_select(0, (torch.LongTensor(adjusted_lengths).cuda() - 1)).cuda()
        print(" > last_out")
        print(outputs_last.size())

        # logits = self.softmax(outputs_last)
        logits = self.logsoftmax(outputs_last).cuda()
        print(" > logits")
        print(logits.size())

        return logits

