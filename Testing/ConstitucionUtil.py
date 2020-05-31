import os
import io
import shutil

import Constant
import csv
import re

import numpy as np

from pathlib import Path


_DATASET = Constant.DATA_FOLDER / "Constitucion\\complete_data.csv"
_RESULT = Constant.RESULTS_FOLDER / "Constitucion"
#_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "OutlierDetection"

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


def separateByConcept(data):
    open_concept = []
    gob_concept = []
    for tuple in data:
        if tuple[1] == 'no':
            gob_concept.append(tuple)
        else:
            open_concept.append(tuple)

    return gob_concept, open_concept


def getMeanVector(phrase, embedding):
    sum_vec = np.array([])
    phrase = re.sub('[^0-9a-zA-Záéíóú]+', ' ', phrase.lower())
    phrase = phrase.strip().split()
    num = 0

    for word in phrase:
        try:
            if sum_vec.size == 0:
                sum_vec = embedding[word]
            else:
                sum_vec = sum_vec + embedding[word]

            num += 1
        except KeyError:
            continue


    if(num == 0):
        return np.array([])

    return (sum_vec / num)


def getSortedDataset(embedding):
    data, header = getDataset()

    gob_concept_vectors = {}
    gob_args_vectors = {}
    open_args_vectors = {}
    mode_vectors = {}

    print("Size of dataset: " + str(len(data)))
    for tuple in data:
        topic = tuple[header[0]]
        is_open_concept = tuple[header[1]]

        original_constitutional_concept = tuple[header[2]]
        original_constitutional_concept_vector = getMeanVector(original_constitutional_concept, embedding)

        constitutional_concept = tuple[header[3]]
        constitutional_concept_vector = getMeanVector(constitutional_concept, embedding)

        argument = tuple[header[4]]
        argument_vector = getMeanVector(argument, embedding)

        argument_mode = tuple[header[5]]
        if not argument_mode in mode_vectors.keys():
            mode_vectors[argument_mode] = getMeanVector(argument_mode, embedding)


        if is_open_concept == 'no':
            if not topic in gob_concept_vectors.keys():
                gob_concept_vectors[topic] = {}
                gob_args_vectors[topic] = []

            if not constitutional_concept in gob_concept_vectors[topic].keys():
                gob_concept_vectors[topic][constitutional_concept] = constitutional_concept_vector

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


    return gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors