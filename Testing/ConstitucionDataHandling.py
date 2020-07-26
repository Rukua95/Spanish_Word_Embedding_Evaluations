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

_DATASET_FOLDER = Constant.DATA_FOLDER / "_Constitucion"
_DATASET = _DATASET_FOLDER / "constitucion_data.csv"


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


#####################################################
# RNN Evaluation
#####################################################

def getDataTaskA():
    train_task_A = {}
    dev_task_A = {}
    test_task_A = {}

    with io.open(_DATASET_FOLDER / "task_A_train.txt", 'r') as f:
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

    with io.open(_DATASET_FOLDER / "task_A_dev.txt", 'r') as f:
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

    with io.open(_DATASET_FOLDER / "task_A_test.txt", 'r') as f:
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


def getDataTaskB():
    data_taskB = {}
    file = _DATASET_FOLDER / "task_B_dataset.txt"

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


def getDataTaskC():
    train_task_C = []
    dev_task_C = []
    test_task_C = []

    with io.open(_DATASET_FOLDER / "task_C_train.txt", 'r') as f:
        for line in f:
            tupla = line.strip().split('/')
            mode = tupla[0]
            arg = tupla[1]

            train_task_C.append([arg, mode])

    print("> train_task_C")
    print(len(train_task_C))

    with io.open(_DATASET_FOLDER / "task_C_dev.txt", 'r') as f:
        for line in f:
            tupla = line.strip().split('/')
            mode = tupla[0]
            arg = tupla[1]

            dev_task_C.append([arg, mode])

    print("> dev_task_C")
    print(len(dev_task_C))

    with io.open(_DATASET_FOLDER / "task_C_test.txt", 'r') as f:
        for line in f:
            tupla = line.strip().split('/')
            mode = tupla[0]
            arg = tupla[1]

            test_task_C.append([arg, mode])

    print("> test_task_C")
    print(len(test_task_C))

    return train_task_C, dev_task_C, test_task_C