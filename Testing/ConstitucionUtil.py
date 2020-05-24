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
    #return sortDataset(data, header)

"""
def sortDataset(data_constitucion, header):
    print("header: {}".format(header))
    print("Cantidad de datos: " + str(len(data_constitucion)) + "\n")


    # Separacion entre argumentos con conceptos de gobierno y argumentos de conceptos abiertos
    open_concept_data = {}
    open_concept_list = []
    open_argument_count = 0

    gob_concept_data = {}
    gob_concept_list = []
    gob_argument_count = 0

    for line in data_constitucion:
        topic = line[header[0]]
        original_constitutional_concept = line[header[2]]
        constitutional_concept = line[header[3]]
        argument = line[header[4]]
        argument_mode = line[header[5]]

        if line[header[1]] == 'no':
            gob_argument_count += 1

            if not topic in gob_concept_data.keys():
                gob_concept_data[topic] = {}

            if not constitutional_concept in gob_concept_data[topic].keys():
                gob_concept_data[topic][constitutional_concept] = []
                gob_concept_list.append(topic + " " + constitutional_concept)

            gob_concept_data[topic][constitutional_concept].append([
                argument,
                argument_mode,
            ])

        else:
            open_argument_count += 1

            if not topic in open_concept_data.keys():
                open_concept_data[topic] = {}

            if not constitutional_concept in open_concept_data[topic].keys():
                open_concept_data[topic][constitutional_concept] = []
                open_concept_list.append(topic + " " + constitutional_concept)

            open_concept_data[topic][constitutional_concept].append([
                original_constitutional_concept,
                argument,
                argument_mode,
            ])


    # Separacion de conceptos abiertos clasificados como conceptos de gobierno, conceptos nuevos, otros e inclasificables
    open_concept_with_gob_concept = {}
    open_concept_as_new_concept = {}
    open_concept_as_other = {}
    open_concept_nondescript = {}

    for topic in open_concept_data.keys():
        if not topic in open_concept_with_gob_concept.keys():
            open_concept_with_gob_concept[topic] = {}

        if not topic in open_concept_as_new_concept.keys():
            open_concept_as_new_concept[topic] = {}

        if not topic in open_concept_as_other.keys():
            open_concept_as_other[topic] = {}

        if not topic in open_concept_nondescript.keys():
            open_concept_nondescript[topic] = {}

        for concept in open_concept_data[topic].keys():
            # Conceptos abiertos equivalentes a conceptos de gobierno
            if (topic + " " + concept) in gob_concept_list:
                open_concept_with_gob_concept[topic][concept] = open_concept_data[topic][concept]

            else:
                # Conceptos otros
                if concept == 'Otro':
                    open_concept_as_other[topic][concept] = open_concept_data[topic][concept]

                # Conceptos inclasificables
                elif concept == 'Inclasificable/No corresponde':
                    open_concept_nondescript[topic][concept] = open_concept_data[topic][concept]

                # Conceptos nuevos
                else:
                    open_concept_as_new_concept[topic][concept] = open_concept_data[topic][concept]


    print("Cantidad de datos con conceptos de gobierno: " + str(gob_argument_count))
    for topic in gob_concept_data.keys():
        print(topic + ") cantidad de conceptos: " + str(len(gob_concept_data[topic])))

    gob_concept_list.sort()
    print("\nCantidad conceptos de gobierno: " + str(len(gob_concept_list)))
    for c in gob_concept_list:
        print(" " + c)


    print("\nCantidad de datos con conceptos abiertos: " + str(open_argument_count))
    for topic in open_concept_data.keys():
        print(topic + ") cantidad de conceptos: " + str(len(open_concept_data[topic])))

    open_concept_list.sort()
    print("\nCantidad de conceptos abiertos: " + str(len(open_concept_list)))
    print("Conceptos nuevos")
    cont = 0
    for c in open_concept_list:
        if not c in gob_concept_list:
            print(" " + c)
            cont += 1

    print("Cantidad de conceptos nuevos: " + str(cont))


    total = 0
    count = 0
    print("\nConteo de datos de argumentos abiertos")
    print("\nConceptos abiertos equivalentes a los conceptos originales de gobierno")
    for topic in open_concept_with_gob_concept.keys():
        print(topic + ", count: " + str(len(open_concept_with_gob_concept[topic])))

        for concept in open_concept_with_gob_concept[topic].keys():
            print("  " + concept + " " + str(len(open_concept_with_gob_concept[topic][concept])))
            count += len(open_concept_with_gob_concept[topic][concept])

    print("Cantidad de datos: " + str(count))

    total += count
    count = 0
    print("\nNuevos conceptos")
    for topic in open_concept_as_new_concept.keys():
        print(topic + ", count: " + str(len(open_concept_as_new_concept[topic])))

        for concept in open_concept_as_new_concept[topic].keys():
            print("  " + concept + " " + str(len(open_concept_as_new_concept[topic][concept])))
            count += len(open_concept_as_new_concept[topic][concept])

    print("Cantidad de datos: " + str(count))

    total += count
    count = 0
    print("\nConceptos clasificados como otros")
    for topic in open_concept_as_other.keys():
        print(topic + ", count: " + str(len(open_concept_as_other[topic])))

        for concept in open_concept_as_other[topic].keys():
            print("  " + concept + " " + str(len(open_concept_as_other[topic][concept])))
            count += len(open_concept_as_other[topic][concept])

    print("Cantidad de datos: " + str(count))

    total += count
    count = 0
    print("\nConceptos inclasificables o que no corresponden")
    for topic in open_concept_nondescript.keys():
        print(topic + ", count: " + str(len(open_concept_nondescript[topic])))

        for concept in open_concept_nondescript[topic].keys():
            print("  " + concept + " " + str(len(open_concept_nondescript[topic][concept])))
            count += len(open_concept_nondescript[topic][concept])

    print("Cantidad de datos: " + str(count))

    total += count
    print("Total de datos: " + str(total))

    return gob_concept_data, open_concept_with_gob_concept, open_concept_as_new_concept, open_concept_as_other, open_concept_nondescript
"""
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


def sortDataset(embedding):
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