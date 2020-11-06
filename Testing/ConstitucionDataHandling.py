import numpy as np

import csv
import re
import io

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



"""
Entrega tres dataset, formateados para las distintas tareas
"""
def getSortedDataset():
    data, header = getDataset()

    dict_task_A = {}
    dict_task_B = {}

    print(" > Tamaño del dataset:", str(len(data)))
    count = 0
    count_gob = 0
    count_open = 0
    for tuple in data:
        count += 1

        topic = tuple[header[0]]
        is_open_concept = tuple[header[1]]
        original_constitutional_concept = tuple[header[2]]
        constitutional_concept = tuple[header[3]]
        argument = tuple[header[4]]

        # Se almacenan los argumentos para conceptos de gobierno
        if is_open_concept == 'no':
            count_gob += 1

            # Inicializacion de topicos
            if not topic in dict_task_A.keys():
                dict_task_A[topic] = {}

            # Inicializacion de conceptos
            if not constitutional_concept in dict_task_A[topic].keys():
                dict_task_A[topic][constitutional_concept] = []

            # Guardamos argumento segun concepto y topico
            dict_task_A[topic][constitutional_concept].append(argument)

        # Se almacenan los argumentos para conceptos abiertos
        else:
            count_open += 1

            # Inicializacion de topicos
            if not topic in dict_task_B.keys():
                dict_task_B[topic] = {}

            # Inicializacion de conceptos
            if not constitutional_concept in dict_task_B[topic].keys():
                dict_task_B[topic][constitutional_concept] = []

            # Guardamos argumento segun concepto y topico
            dict_task_B[topic][constitutional_concept].append(original_constitutional_concept)

    print(" > Conceptos de gobierno")
    for topic in dict_task_A.keys():
        print("Topico", topic, "=> cantidad de conceptos: ", len(dict_task_A[topic].keys()))

    print(" > Conceptos abiertos:")
    for topic in dict_task_B.keys():
        print("Topico", topic, "=> cantidad de conceptos: ", len(dict_task_B[topic].keys()))

    # Se eliminan conceptos abiertos que no se clasifican en originales
    print(" > Conceptos abiertos eliminado")
    for topic in dict_task_B.keys():
        delete_concept = []
        for concept in dict_task_B[topic].keys():
            if concept not in dict_task_A[topic].keys():
                delete_concept.append(concept)

        print("Topico", topic, "=> cantidad de conceptos eliminado: ", len(delete_concept))
        for concept in delete_concept:
            dict_task_B[topic].pop(concept, None)

    print(" > Cantidad de argumentos para conceptos de gobierno:", count_gob)
    print(" > Cantidad de argumentos para conceptos abiertos:", count_open)

    return dict_task_A, dict_task_B



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

