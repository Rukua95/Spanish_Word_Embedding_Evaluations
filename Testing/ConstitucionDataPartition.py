import ConstitucionUtil
import re
import random

data, header = ConstitucionUtil.getDataset()
gob_concepts = {}
gob_concepts_count = 0
open_concepts = {}
open_concepts_count = 0

print("total: " + str(len(data)))
for tuple in data:
    # re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[0]].lower())
    topic = re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[0]].lower())
    is_open_concept = re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[1]].lower())

    original_constitutional_concept = re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[2]].lower())
    constitutional_concept = re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[3]].lower())

    argument = re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[4]].lower())
    argument_mode = re.sub('[^0-9a-zA-Záéíóú]+', ' ', tuple[header[5]].lower())

    if is_open_concept == 'no':
        gob_concepts_count += 1
        if topic not in gob_concepts.keys():
            gob_concepts[topic] = []

        gob_concepts[topic].append([original_constitutional_concept, constitutional_concept, argument, argument_mode])

    else:
        open_concepts_count += 1
        if topic not in open_concepts.keys():
            open_concepts[topic] = []

        open_concepts[topic].append([original_constitutional_concept, constitutional_concept, argument, argument_mode])

print("\ngob_concept " + str(gob_concepts_count))
for topic in gob_concepts.keys():
    print("topic: " + topic, end=' ')
    print("len: " + str(len(gob_concepts[topic])))
    for line in gob_concepts[topic][:5]:
        print(line)

print("\nopen_concept " + str(open_concepts_count))
for topic in open_concepts.keys():
    print("topic: " + topic)
    print("len: " + str(len(open_concepts[topic])))
    for line in open_concepts[topic][:5]:
        print(line)

# %% md

Separacion
de
dataset

# %%

# Generar dataset train, dev y test, y guardarlos

import Constant
import io

_DATASET = Constant.DATA_FOLDER / "_Constitucion"

taskA_train = _DATASET / "task_A_train.txt"
taskA_dev = _DATASET / "task_A_dev.txt"
taskA_test = _DATASET / "task_A_test.txt"

"""
Task A
Separacion en train, dev y test sets
 - 
"""

good_tuples = {}
count_by_mode = {}
lista_conceptos = {}
good_tuples_count = 0
for topic in gob_concepts.keys():
    if topic not in good_tuples.keys():
        good_tuples[topic] = {}
        lista_conceptos[topic] = []

    for tupla in gob_concepts[topic]:
        if tupla[3] not in count_by_mode.keys():
            count_by_mode[tupla[3]] = 0

        if tupla[3] == "blank":  # or tupla[3] == "undefined":
            continue

        count_by_mode[tupla[3]] += 1
        good_tuples_count += 1
        concept = tupla[1]

        if concept not in [topic]:
            lista_conceptos[topic].append(concept)

        if concept not in good_tuples[topic].keys():
            good_tuples[topic][concept] = []

        good_tuples[topic][concept].append(tupla)

print(good_tuples_count)
for mode in count_by_mode.keys():
    print(mode + " " + str(count_by_mode[mode]))

train_set = {}
dev_set = {}
test_set = {}

for topic in good_tuples.keys():
    print(topic + " " + str(len(good_tuples[topic].keys())))
    train_set[topic] = {}
    dev_set[topic] = {}
    test_set[topic] = {}

    for concept in good_tuples[topic].keys():
        print(" > " + concept + " " + str(len(good_tuples[topic][concept])))
        train_set[topic][concept] = []
        dev_set[topic][concept] = []
        test_set[topic][concept] = []

        N = len(good_tuples[topic][concept])

        n_train = int(N * 0.8)
        n_dev = int((N - n_train) * 0.5)
        n_test = N - n_train - n_dev

        concep_test_set = []
        concep_dev_set = []
        concep_train_set = []

        aux_set = []
        index = range(N)
        train_index = random.sample(index, n_train)
        for i in range(N):
            if i in train_index:
                concep_train_set.append(good_tuples[topic][concept][i])
            else:
                aux_set.append(good_tuples[topic][concept][i])

        index = range(len(aux_set))
        dev_index = random.sample(index, n_dev)
        for i in range(len(aux_set)):
            if i in dev_index:
                concep_dev_set.append(aux_set[i])
            else:
                concep_test_set.append(aux_set[i])

        print("   " + str(len(concep_train_set)) + " " + str(len(concep_dev_set)) + " " + str(len(concep_test_set)))

        train_set[topic][concept] = concep_train_set
        dev_set[topic][concept] = concep_dev_set
        test_set[topic][concept] = concep_test_set


# Guardar dataset para task A
def guardar(save_file, content):
    with io.open(save_file, 'w') as f:
        for topic in content.keys():
            for concept in content[topic].keys():
                for tupla in content[topic][concept]:
                    f.write(topic + "/" + concept + "/" + tupla[2] + "\n")


guardar(taskA_train, train_set)
guardar(taskA_dev, dev_set)
guardar(taskA_test, test_set)

# %%

"""
Task B
Guardar todos los elementos asociados a conceptos de gobierno por topic, concept_gob, argumento + concepto_original
"""

taskB_data = _DATASET / "task_B_dataset.txt"

data = {}
data_count = 0
for topic in open_concepts.keys():
    print(topic)
    if topic not in data.keys():
        data[topic] = {}

        for i in range(5):
            print(open_concepts[topic][i])

    for tupla in open_concepts[topic]:
        orig_concept = tupla[0]
        gob_concept = tupla[1]
        argument = tupla[2]
        arg_mode = tupla[3]

        # if arg_mode == 'blank':# or arg_mode == 'undefined':
        #    continue

        if gob_concept not in lista_conceptos[topic]:
            continue

        if gob_concept not in data[topic].keys():
            data[topic][gob_concept] = []

        data_count += 1
        data[topic][gob_concept].append([orig_concept, argument])

with io.open(taskB_data, 'w') as f:
    print("total " + str(data_count))

    for topic in data.keys():
        print(topic + " " + str(len(data[topic].keys())))

        for concept in data[topic].keys():
            print(" > " + concept + " " + str(len(data[topic][concept])))

            for tupla in data[topic][concept]:
                f.write(topic + "/" + concept + "/" + tupla[0] + "/" + tupla[1] + "\n")

# %%

"""
Task C
tomar todos los argumentos, eliminar blanks y undefined, y separarlos segun modo de argumentacion
"""

arg_data = {}
arg_count = 0
for topic in gob_concepts:
    for tupla in (gob_concepts[topic] + open_concepts[topic]):
        argument = tupla[2]
        arg_mode = tupla[3]

        if arg_mode == 'blank' or arg_mode == 'undefined':
            continue

        if arg_mode not in arg_data.keys():
            arg_data[arg_mode] = []

        arg_data[arg_mode].append(argument)
        arg_count += 1

print("total " + str(arg_count))
for mode in arg_data.keys():
    print(" > " + mode + " " + str(len(arg_data[mode])))

    for i in range(3):
        print(arg_data[mode][i])

taskC_train = _DATASET / "task_C_train.txt"
taskC_dev = _DATASET / "task_C_dev.txt"
taskC_test = _DATASET / "task_C_test.txt"

taskC_train_set = []
taskC_dev_set = []
taskC_test_set = []

with io.open(taskC_train, 'w') as f:
    for mode in arg_data.keys():

        for arg in arg_data[mode]:
            f.write(mode + "/" + str(arg) + "\n")