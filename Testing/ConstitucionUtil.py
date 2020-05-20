import os
import io
import shutil

import Constant

import numpy as np

from pathlib import Path


_DATASET = Constant.DATA_FOLDER / "Constitucion\\complete_data.csv"
_RESULT = Constant.RESULTS_FOLDER / "Constitucion"
#_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "OutlierDetection"

def getDataset():
    data = []
    with io.open(_DATASET, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(',')

            data.append(line)

    data.sort()
    return data


def separateByConcept(data):
    open_concept = []
    gob_concept = []
    for tuple in data:
        if tuple[1] == "no":
            gob_concept.append(tuple)
        else:
            open_concept.append(tuple)

    return gob_concept, open_concept


def separataByTopic(data):
    topics = [[], [], [], []]
    for tuple in data:
        for i in range(4):
            if str(tuple[0]) == str(i + 1):
                topics[i].append(tuple)
                continue

    return topics


def getMeanVector(phrase, embedding):
    sum_vec = 0
    phrase = phrase.strip().split()
    num = 0

    for word in phrase:
        try:
            if sum_vec == 0:
                sum_vec = embedding[word]
            else:
                sum_vec += embedding[word]

            num += 1
        except KeyError:
            continue


    if(num == 0):
        return 0

    return (sum_vec / num)



