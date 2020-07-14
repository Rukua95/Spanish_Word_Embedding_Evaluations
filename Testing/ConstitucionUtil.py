import csv
import re


import GPUtil

import torch
import torch.nn as nn

from random import shuffle

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity

from pytorchtools import EarlyStopping


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

        self.embedding.cuda()

        self.lstm = nn.LSTM(
            input_size=emb_weight.size()[1],
            hidden_size=label_size,
            bidirectional=False,
            batch_first=True)

        self.lstm.cuda()

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, entity_ids, seq_len):
        max_seq_len = torch.max(seq_len)

        emb = self.embedding(entity_ids)

        out, _ = self.lstm(emb)

        out = out.reshape(out.size()[0] * out.size()[1], out.size()[2])

        adjusted_lengths = [i * max_seq_len + l for i, l in enumerate(seq_len)]
        outputs_last = out.index_select(0, (torch.LongTensor(adjusted_lengths).cuda() - 1))

        # logits = self.softmax(outputs_last)
        logits = self.logsoftmax(outputs_last)

        return logits

class RNNEvaluation():
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _lower = True
    _oov_word = {}
    _batch_size = 256

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "_Constitucion"
    _RESULT = Constant.RESULTS_FOLDER / "Constitucion_rnn"
    MODEL_FOLDER = Constant.MAIN_FOLDER / "Models"


    def __init__(self, cantidad=None, batch_size=512, lower=True):
        print("Test de Constitucion")

        self._embeddings_size = cantidad
        self._batch_size = batch_size
        self._lower = lower

        if not self.MODEL_FOLDER.exists():
            os.makedirs(self.MODEL_FOLDER)


    def getDataTaskA(self):
        train_task_A = {}
        dev_task_A = {}
        test_task_A = {}

        with io.open(self._DATASET / "task_A_train.txt", 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                topic = tupla[0]
                gob_concept = tupla[1]
                argument = tupla[2]

                if topic not in train_task_A:
                    train_task_A[topic] = []

                train_task_A[topic].append([argument, gob_concept])

        print("> train_task_A")
        for topic in train_task_A.keys():
            print(topic, str(len(train_task_A[topic])))

        with io.open(self._DATASET / "task_A_dev.txt", 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                topic = tupla[0]
                gob_concept = tupla[1]
                argument = tupla[2]

                if topic not in dev_task_A:
                    dev_task_A[topic] = []

                dev_task_A[topic].append([argument, gob_concept])

        print("> dev_task_A")
        for topic in dev_task_A.keys():
            print(topic, str(len(dev_task_A[topic])))

        with io.open(self._DATASET / "task_A_test.txt", 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                topic = tupla[0]
                gob_concept = tupla[1]
                argument = tupla[2]

                if topic not in test_task_A:
                    test_task_A[topic] = []

                test_task_A[topic].append([argument, gob_concept])

        print("> test_task_A")
        for topic in test_task_A.keys():
            print(topic, str(len(test_task_A[topic])))

        return train_task_A, dev_task_A, test_task_A


    def getDataTaskB(self):
        data_taskB = {}
        file = self._DATASET / "task_B_dataset.txt"

        with io.open(file, 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                topic = tupla[0]
                gob_concept = tupla[1]
                open_concept = tupla[2]
                argument = tupla[3]

                if topic not in data_taskB:
                    data_taskB[topic] = []

                data_taskB[topic].append([argument, open_concept, gob_concept])

        print("> data_taskB")
        for topic in data_taskB.keys():
            print(topic, str(len(data_taskB[topic])))

        return data_taskB


    def getDataTaskC(self):
        train_task_C = []
        dev_task_C = []
        test_task_C = []

        with io.open(self._DATASET / "task_C_train.txt", 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                mode = tupla[0]
                arg = tupla[1]

                train_task_C.append([arg, mode])

        print("> train_task_C")
        print(len(train_task_C))

        with io.open(self._DATASET / "task_C_dev.txt", 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                mode = tupla[0]
                arg = tupla[1]

                dev_task_C.append([arg, mode])

        print("> dev_task_C")
        print(len(dev_task_C))

        with io.open(self._DATASET / "task_C_test.txt", 'r') as f:
            for line in f:
                tupla = line.strip().split('/')
                mode = tupla[0]
                arg = tupla[1]

                test_task_C.append([arg, mode])

        print("> test_task_C")
        print(len(test_task_C))

        return train_task_C, dev_task_C, test_task_C

    def cleanDataVocab(self, data, word_vector, replace_oov=False):
        print(">>> CleanDataVocab")
        revised_data = {}

        for key in data.keys():
            revised_data[key] = []
            new_pair = []

            for pair in data[key]:
                for i in range(len(pair) - 1):
                    try:
                        l = pair[i].strip().split()
                        r = []
                    except:
                        print(pair)
                        raise Exception

                    for word in l:
                        if word not in word_vector:
                            if replace_oov:
                                word_vector.add(word, np.random.rand(word_vector.vector_size))
                            else:
                                continue

                        r.append(word)

                    if len(r) == 0:
                        new_pair = []
                        break

                    new_pair.append(r)

                if len(new_pair) == 0:
                    continue

                new_pair.append(pair[-1])
                revised_data[key].append(new_pair)

                new_pair = []

            print(key, str(len(revised_data[key])))
            #del word_vector.vectors_norm

        return revised_data

    def padding(self, batch):
        pad = "<pad>"
        args = [x[0] for x in batch]
        cons = [x[1] for x in batch]

        seq_lengths = list(map(len, args))
        max_lengths = max(seq_lengths)
        for i in range(len(args)):
            args[i] = (args[i] + [pad for i in range(max_lengths - len(args[i]))]) if len(args[i]) < max_lengths else args[i]

        return [args, seq_lengths, cons]

    def generateBatch(self, data, batch_size):
        shuffle(data)
        batches = []
        aux_batch = []
        for pair in data:
            argument = pair[0]
            concept = pair[1]

            if argument == []:
                continue

            aux_batch.append([argument, concept])
            if len(aux_batch) == batch_size:
                batches.append(aux_batch)
                aux_batch = []

        if len(aux_batch) != 0:
            batches.append(aux_batch)

        for i in range(len(batches)):
            args, len_seq, cons = self.padding(batches[i])
            batches[i] = [args, len_seq, cons]

        return batches

    def line2vecs(self, word_vector, arguments):
        tensor = torch.zeros([len(arguments), len(arguments[0])], dtype=torch.long)

        for j in range(len(arguments)):
            arg = arguments[j]
            for i in range(len(arg)):
                word = arg[i]
                if word not in word_vector and word != "<pad>":
                    print(" word not found: " + word)
                    word_vector.add(word, np.random.rand(word_vector.vector_size))

                if word == "<pad>":
                    tensor[j][i] = 0
                else:
                    # t = torch.zeros(1, len(word_vector.vocab))
                    # t[0][word_vector.vocab[word].index] = 1
                    tensor[j][i] = word_vector.vocab[word].index + 1

        return tensor


    def getTrainExample(self, arguments, concepts, concepts_list, word_vector):
        argument_vectors = self.line2vecs(word_vector, arguments)
        label_vector = torch.tensor(
            [torch.tensor([concepts_list.index(concept)], dtype=torch.long) for concept in concepts])

        return argument_vectors, label_vector


    def results(self, prediction, concept):
        predict_idx = (torch.topk(prediction, 5)).indices
        predict_idx = predict_idx.cpu()

        top1 = 0
        top5 = 0

        for i in range(len(predict_idx)):
            if concept[i] == predict_idx[i][0]:
                top1 += 1

            if concept[i] in predict_idx[i]:
                top5 += 1

        return top1, top5


    def trainAndTestTaskA(self, word_vector, word_vector_name):
        # Preparacion
        print("inicio")
        GPUtil.showUtilization()
        resultsA = {}
        resultsB = {}

        train, dev, test = self.getDataTaskA()

        print(">>> Limpiando train data")
        clean_train = self.cleanDataVocab(train, word_vector)

        print(">>> Limpiando dev data")
        clean_dev = self.cleanDataVocab(dev, word_vector)

        print(">>> Limpiando test data")
        clean_test = self.cleanDataVocab(test, word_vector)

        pad = torch.FloatTensor([np.random.rand(word_vector.vector_size)])
        for topic in clean_train.keys():
            print(topic + " " + str(len(clean_train[topic])) + " " + str(len(train[topic])))

            for i in range(5):
                print(" > ", end='')
                print(clean_train[topic][i])

        weight = torch.FloatTensor(word_vector.vectors)
        weight = torch.cat([pad, weight])

        criterion = nn.NLLLoss()

        for topic in clean_train.keys():
            print("Training for topic", topic)
            GPUtil.showUtilization()

            train_data = clean_train[topic]
            dev_data = clean_dev[topic]
            test_data = clean_test[topic]

            print("Amount of pairs: " + str(len(train_data)) + "\n")

            concept_list = []
            for pair in train_data:
                concept = pair[1]
                if concept not in concept_list:
                    concept_list.append(concept)

            concept_list.sort()
            n_output = len(concept_list)
            print("num_concept: " + str(n_output))
            print(concept_list)

            # Get RNN
            mylstm = ClassifierModel(n_output, weight)
            optimizer = torch.optim.SGD(mylstm.parameters(), lr=0.001)
            print(mylstm)

            epoch = 200
            batch_size = self._batch_size

            train_losses = []
            valid_losses = []
            avg_train_losses = []
            avg_valid_losses = []

            early_save = self.MODEL_FOLDER / ("taskA_" + str(topic) + "_" + word_vector_name + ".pt")
            using_last_save = False
            if early_save.exists():
                mylstm.load_state_dict(torch.load(early_save))
                using_last_save = True

            early_stopping = EarlyStopping(patience=10, verbose=True, path=early_save)

            for ep in range(1, epoch + 1):
                print("Epoca: " + str(ep))
                GPUtil.showUtilization()

                # Generate batchs
                train_batch = self.generateBatch(train_data, batch_size)
                dev_batch = self.generateBatch(dev_data, batch_size)

                mylstm.train()
                print(">>> Test: Number of batchs:", str(len(train_batch)))
                for b in train_batch:
                    if using_last_save:
                        using_last_save = False
                        break

                    arg = b[0]
                    size = b[1]
                    con = b[2]

                    with torch.no_grad():
                        arg, con = self.getTrainExample(arg, con, concept_list, word_vector)
                        size = torch.tensor(size, dtype=torch.long)

                    mylstm.zero_grad()
                    optimizer.zero_grad()

                    output = mylstm(arg.cuda(), size.cuda())
                    del arg
                    del size
                    torch.cuda.empty_cache()

                    loss = criterion(output, con.cuda())
                    del con
                    del output
                    torch.cuda.empty_cache()

                    loss.backward()

                    optimizer.step()

                    train_losses.append(loss.item())
                    del loss
                    torch.cuda.empty_cache()


                mylstm.eval()
                print(">>> Validation: Number of batchs:", str(len(train_batch)))
                GPUtil.showUtilization()
                for b in dev_batch:
                    arg = b[0]
                    size = b[1]
                    con = b[2]

                    with torch.no_grad():
                        arg, con = self.getTrainExample(arg, con, concept_list, word_vector)
                        size = torch.tensor(size, dtype=torch.long)

                    output = mylstm(arg.cuda(), size.cuda())
                    del arg
                    del size
                    torch.cuda.empty_cache()

                    loss = criterion(output, con.cuda())
                    del output
                    del con
                    torch.cuda.empty_cache()

                    valid_losses.append(loss.item())
                    del loss
                    torch.cuda.empty_cache()

                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                print("Epoca: " + str(ep))
                print("Train loss", train_loss)
                print("Valid loss", valid_loss)

                # clear lists to track next epoch
                train_losses = []
                valid_losses = []
                early_stopping(valid_loss, mylstm)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # load the last checkpoint with the best model
            mylstm.load_state_dict(torch.load(early_save))
            t1, t5 = self.testTaskA(mylstm, test_data, concept_list, word_vector)
            resultsA[topic] = [t1, t5]

            torch.cuda.empty_cache()

            # Aprovechamos de obtener resultados para task B
            data_task_B = self.getDataTaskB()
            clean_data_task_B = self.cleanDataVocab(data_task_B, word_vector)
            concat_t1, concat_t5, concept_t1, concept_t5 = self.testTaskB(mylstm, clean_data_task_B[topic], concept_list, word_vector)
            resultsB[topic] = [concat_t1, concat_t5, concept_t1, concept_t5]

            del mylstm
            torch.cuda.empty_cache()

        return resultsA, resultsB


    def trainAndTestTaskC(self, word_vector, word_vector_name):
        # Preparacion
        print("inicio")
        GPUtil.showUtilization()
        resultsC = {}

        train, dev, test = self.getDataTaskC()

        print(">>> Limpiando train data")
        clean_train = self.cleanDataVocab({"m": train}, word_vector)

        print(">>> Limpiando dev data")
        clean_dev = self.cleanDataVocab({"m": dev}, word_vector)

        print(">>> Limpiando tes data")
        clean_test = self.cleanDataVocab({"m": test}, word_vector)

        pad = torch.FloatTensor([np.random.rand(word_vector.vector_size)])
        for topic in clean_train.keys():
            print(topic, str(len(clean_train[topic])), str(len(train)))

            for i in range(5):
                print(" > ", end='')
                print(clean_train[topic][i])

        weight = torch.FloatTensor(word_vector.vectors)
        weight = torch.cat([pad, weight])

        criterion = nn.NLLLoss()

        for topic in clean_train.keys():
            print("Preparando entrenamiento")
            GPUtil.showUtilization()

            train_data = clean_train[topic]
            dev_data = clean_dev[topic]
            test_data = clean_test[topic]

            print("Amount of pairs: " + str(len(train_data)) + "\n")

            mode_list = []
            for pair in train_data:
                concept = pair[1]
                if concept not in mode_list:
                    mode_list.append(concept)

            mode_list.sort()
            n_output = len(mode_list)
            print("Num_mode: " + str(n_output))
            print(mode_list)

            # Get RNN
            mylstm = ClassifierModel(n_output, weight)
            optimizer = torch.optim.SGD(mylstm.parameters(), lr=0.001)
            print(mylstm)

            epoch = 200
            batch_size = self._batch_size

            train_losses = []
            valid_losses = []
            avg_train_losses = []
            avg_valid_losses = []

            early_save = self.MODEL_FOLDER / ("taskC_" + word_vector_name + ".pt")
            early_stopping = EarlyStopping(patience=10, verbose=True, path=early_save)

            using_last_save = False
            if early_save.exists():
                mylstm.load_state_dict(torch.load(early_save))
                using_last_save = True

            for ep in range(1, epoch + 1):
                print("Epoca: " + str(ep))
                GPUtil.showUtilization()

                # Generate batchs
                train_batch = self.generateBatch(train_data, batch_size)
                dev_batch = self.generateBatch(dev_data, batch_size)

                mylstm.train()
                print(">>> Training task C: Number of batches:", str(len(train_batch)))
                for b in train_batch:
                    if using_last_save:
                        using_last_save = False
                        break

                    arg = b[0]
                    size = b[1]
                    con = b[2]

                    with torch.no_grad():
                        arg, con = self.getTrainExample(arg, con, mode_list, word_vector)
                        size = torch.tensor(size, dtype=torch.long)

                    mylstm.zero_grad()
                    optimizer.zero_grad()

                    output = mylstm(arg.cuda(), size.cuda())
                    del arg
                    del size
                    torch.cuda.empty_cache()

                    loss = criterion(output, con.cuda())
                    del con
                    del output
                    torch.cuda.empty_cache()

                    loss.backward()

                    optimizer.step()

                    train_losses.append(loss.item())
                    del loss
                    torch.cuda.empty_cache()


                mylstm.eval()
                print(">>> Validation task C: Number of batches:", str(len(dev_batch)))
                GPUtil.showUtilization()
                for b in dev_batch:
                    arg = b[0]
                    size = b[1]
                    con = b[2]

                    with torch.no_grad():
                        arg, con = self.getTrainExample(arg, con, mode_list, word_vector)
                        size = torch.tensor(size, dtype=torch.long)

                    output = mylstm(arg.cuda(), size.cuda())
                    del arg
                    del size
                    torch.cuda.empty_cache()

                    loss = criterion(output, con.cuda())
                    del output
                    del con
                    torch.cuda.empty_cache()

                    valid_losses.append(loss.cpu().item())
                    del loss
                    torch.cuda.empty_cache()

                train_loss = np.average(train_losses)
                valid_loss = np.average(valid_losses)
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                print("Epoca: " + str(ep))
                print("Train loss", train_loss)
                print("Valid loss", valid_loss)

                # clear lists to track next epoch
                train_losses = []
                valid_losses = []
                early_stopping(valid_loss, mylstm)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            # load the last checkpoint with the best model
            mylstm.load_state_dict(torch.load(early_save))
            p, r, f1 = self.testTaskC(mylstm, test_data, mode_list, word_vector)
            resultsC[topic] = [p, r, f1]

            del mylstm
            torch.cuda.empty_cache()

            print("fin de todo")
            GPUtil.showUtilization()

        return resultsC


    def testTaskA(self, mylstm, test_data, concept_list, word_vector):
        print("Test A")
        GPUtil.showUtilization()

        batch_size = self._batch_size
        test_batch = self.generateBatch(test_data, batch_size)
        total = len(test_data)

        total_top1 = 0
        total_top5 = 0

        count = 0
        mylstm.eval()
        for b in test_batch:
            print("Batch")
            GPUtil.showUtilization()

            with torch.no_grad():
                arg = b[0]
                size = b[1]
                con = b[2]

            arg, con = self.getTrainExample(arg, con, concept_list, word_vector)
            size = torch.tensor(size, dtype=torch.long)

            output = mylstm(arg.cuda(), size.cuda())
            del arg
            del size
            torch.cuda.empty_cache()

            top1, top5 = self.results(output, con.cuda())
            del output
            del con
            torch.cuda.empty_cache()

            total_top1 += top1
            total_top5 += top5

        top1 = total_top1 / total
        top5 = total_top5 / total
        print("total_top1", top1)
        print("total_top5", top5)

        return top1, top5



    def testTaskB(self, mylstm, data_task_B, concept_list, word_vector):
        print("Test B")
        GPUtil.showUtilization()

        batch_size = self._batch_size
        total = len(data_task_B)

        print("Concat data")
        # Reordenar data
        concat_B_task_data = [[d[0] + d[1], d[2]] for d in data_task_B]

        # Generar batchs
        batches_arg_con = self.generateBatch(concat_B_task_data, batch_size)

        concat_total_top1 = 0
        concat_total_top5 = 0

        count = 0
        mylstm.eval()
        for b in batches_arg_con:
            print("Batch")
            GPUtil.showUtilization()

            with torch.no_grad():
                arg = b[0]
                size = b[1]
                con = b[2]

            arg, con = self.getTrainExample(arg, con, concept_list, word_vector)
            size = torch.tensor(size, dtype=torch.long)

            output = mylstm(arg.cuda(), size.cuda())
            del arg
            del size
            torch.cuda.empty_cache()

            top1, top5 = self.results(output, con.cuda())
            del output
            del con
            torch.cuda.empty_cache()

            concat_total_top1 += top1
            concat_total_top5 += top5

        concat_total_top1 = concat_total_top1 / total
        concat_total_top5 = concat_total_top5 / total


        print("Only concept data")
        # Reordenar data
        concep_B_task_data = [[d[1], d[2]] for d in data_task_B]

        # Generar batchs
        batches_con = self.generateBatch(concep_B_task_data, batch_size)

        concep_total_top1 = 0
        concep_total_top5 = 0

        mylstm.eval()
        for b in batches_con:
            print("Batch")
            GPUtil.showUtilization()

            with torch.no_grad():
                arg = b[0]
                size = b[1]
                con = b[2]

            arg, con = self.getTrainExample(arg, con, concept_list, word_vector)
            size = torch.tensor(size, dtype=torch.long)

            output = mylstm(arg.cuda(), size.cuda())
            del arg
            del size
            torch.cuda.empty_cache()

            top1, top5 = self.results(output, con.cuda())
            del output
            del con
            torch.cuda.empty_cache()

            concep_total_top1 += top1
            concep_total_top5 += top5

        concep_total_top1 = concep_total_top1 / total
        concep_total_top5 = concep_total_top5 / total

        print("Test B result")
        print(concat_total_top1, concat_total_top5, concep_total_top1, concep_total_top5)

        return concat_total_top1, concat_total_top5, concep_total_top1, concep_total_top5


    def testTaskC(self, mylstm, test_data, mode_list, word_vector):
        print("Test C")
        GPUtil.showUtilization()

        batch_size = self._batch_size
        test_batch = self.generateBatch(test_data, batch_size)

        prediction = []
        label = []

        mylstm.eval()
        for b in test_batch:
            print("Batch")
            GPUtil.showUtilization()

            with torch.no_grad():
                arg = b[0]
                size = b[1]
                con = b[2]

            arg, con = self.getTrainExample(arg, con, mode_list, word_vector)
            size = torch.tensor(size, dtype=torch.long)

            output = mylstm(arg.cuda(), size.cuda())
            del arg
            del size
            torch.cuda.empty_cache()

            pred = (torch.topk(output, 1)).indices
            pred = torch.reshape(pred.cpu(), (-1,))

            prediction.append(pred)
            label.append(con)
            del output
            del pred
            torch.cuda.empty_cache()

        prediction = torch.cat(prediction).numpy()
        label = torch.cat(label).numpy()

        res = precision_recall_fscore_support(label, prediction, labels=np.array(range(len(mode_list))), average='macro')

        print("fin de test")
        GPUtil.showUtilization()

        print(res)
        return res

    def saveResults(self, word_vector_name , result_taskA, result_taskB, result_taskC):
        save_path = self._RESULT

        if not save_path.exists():
            os.makedirs(save_path)

        result_path = save_path / ("rnn_" + word_vector_name + ".txt")

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

                f.write("Concat Top1 " + str(tupla[0]) + " Top5 " + str(tupla[1]) + "\n")
                f.write("Concept Top1 " + str(tupla[2]) + " Top5 " + str(tupla[3]) + "\n")

            f.write("Task C results\n")
            for key in result_taskC.keys():
                tupla = result_taskC[key]

                f.write("Presicion " + str(tupla[2]) + " Recall " + str(tupla[3]) + " F1 " + str(
                    2 / (1 / tupla[2] + 1 / tupla[3])) + "\n")


    def evaluate(self):
        # Iterar por todos los embeddings
        # Realizacion de test por cada embedding
        print("\n>>> Inicio de test <<<\n")
        for embedding_name in self._embeddings_name_list:
            word_vector_name = embedding_name.split('.')[0]
            word_vector = get_wordvector(embedding_name, self._embeddings_size)

            # Task A y B
            resA, resB = self.trainAndTestTaskA(word_vector, word_vector_name)

            # Task C
            resC = self.trainAndTestTaskC(word_vector, word_vector_name)

            # Guardar resultados
            self.saveResults(word_vector_name, resA, resB, resC)
