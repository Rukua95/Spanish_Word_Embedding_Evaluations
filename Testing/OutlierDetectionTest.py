import os
import io
import shutil

import Constant

import numpy as np

from pathlib import Path


_DATASET = Constant.DATA_FOLDER / "OutlierDetectionDataset"
_RESULT = Constant.RESULTS_FOLDER / "OutlierDetection"
_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "OutlierDetection"

###########################################################################################
# METRICAS
###########################################################################################

"""
Retorna Pseudo-Inverted Compactness Score (Camacho-Colados) para todos los w en el conjunto C = W + {w} 
"""
def getPseudoInvertedCompactnessScore(embedding, W, w, phrase):
    sum = 0
    k = 0
    for word in W:
        # TODO: inclusion de frase durante la validacion

        sum += (embedding.similarity(word, w) + embedding.similarity(w, word))
        k += 1

    return (sum / k)


"""
Obtencion de puntaje del test para un solo archivo/conjunto

:param embedding: lista de vectores de palabras
:param main_set: conjunto principal de palabras
:param outlier_set: conjunto de palabras outliers
:return: valor de OP y OD para cada palabra en outlier_set, respecto a main_set
"""
def getFileScores(embedding, main_set, outlier_set, phrase, existe_oov):
    # TODO: manejo de palabras fuera del vocabulario

    OP = []
    OD = []
    for outlier in outlier_set:
        # Obtencion de puntaje para outlier (nos importa su posicion respecto al resto de puntajes)
        W = main_set
        p = getPseudoInvertedCompactnessScore(embedding, W, outlier, phrase)
        pos = len(main_set)

        W.append(outlier)
        for i in range(len(W)):
            if W[i] == outlier:
                continue

            C = W
            w = W[i]
            C.pop(i)

            print(C)

            p_i = getPseudoInvertedCompactnessScore(embedding, C, w)
            if p_i < p:
                pos -= 1

        OP.append(pos)
        OD.append(1 if pos == len(main_set) else 0)

    return OP, OD


"""
Obtencion de accuracy y OPP

:param embedding: lista de vectores de palabras
:param test_sets: lista de pares de conjuntos [main_set, outlier_set]
:return: accuracy y OPP
"""
def getScores(embedding, test_sets, phrase, existe_oov):
    cant_test = 0
    sum_op = 0.0
    sum_od = 0.0
    for test in test_sets:
        main_set, outlier_set = test

        OP_list, OD_list = getFileScores(embedding, main_set, outlier_set, phrase, existe_oov)
        temp_sum_op = 0.0
        temp_sum_od = 0.0
        for op in OP_list:
            temp_sum_op += op

        sum_op += (temp_sum_op / len(main_set))

        for od in OD_list:
            temp_sum_od += od

        sum_od += temp_sum_od

        cant_test += len(outlier_set)

    accuraccy = (sum_op / cant_test)
    OPP = (sum_od / cant_test)

    return accuraccy, OPP





###########################################################################################
# MANEJO DE ARCHIVOS
###########################################################################################

"""
Obtencion de la lista de archivos test
"""
def getTestFiles():
    if _DATASET.exists():
        raise Exception("No se logro encontrar carpeta con test")

    return os.listdir(_DATASET)


"""
Obtencion de las palabras desde el archivo de test, palabras del conjunto y conunto outlier
"""
def getWords(file_name):
    main_set = []
    outlier_set = []

    with io.open(_DATASET / file_name) as f:
        for line in f:
            if line == "\n":
                main_set = outlier_set
                outlier_set = []
                continue

            line = line.strip()
            outlier_set.append(line)

    return main_set, outlier_set


###########################################################################################
# GUARDAR RESULTADOS
###########################################################################################

"""
Guardado de resultados del test por outlier detection

:param embedding_name: nombre de embedding evaluado
:param results: lista de los resultados (accuraccy, OPP) y datos (% oov outlier, % ovv main set, % grupos omitidos)
                de un embedding, la lista se compone de pares [nombre_resultado, valor_resultado]
"""
def saveResults(embedding_name, results, phrase, existe_oov):
    extension = ""
    if phrase:
        extension += "_phrase"

    if not existe_oov:
        extension += "_vocabIntersect"

    save_path = _RESULT / (embedding_name + extension + ".txt")
    with io.open(save_path, 'w') as f:
        for r in results:
            f.write(r[0] + " " + str(r[1]) + "\n")


###########################################################################################
# EVALUACION POR OUTLIER DETECTION
###########################################################################################

"""
Realizacion de outlier detection test
"""
def outlierDetectionTest(embedding, embedding_name, phrase=False, existe_oov=True):
    # Obtencion de conjuntos
    file_test_list = getTestFiles()
    pair_test_list = []
    for file in file_test_list:
        pair_test_list.append(getWords(file))

    # TODO: adicion de datos (%oov outlier, %oov main set, %grupos omitidos)
    score = getScores(embedding, pair_test_list, phrase, existe_oov)
    results = [["accuraccy", score[0]], ["OPP", score[1]]]

    saveResults(embedding_name, results)