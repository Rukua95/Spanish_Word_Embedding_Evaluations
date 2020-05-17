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
# PUNTUACION
###########################################################################################


"""
Retorna Pseudo-Inverted Compactness Score (Camacho-Colados) para todos los w en el conjunto C = W + {w}

:param embedding: lista de vectores de palabras
:param W: lista de palabras
:param w: palabra a la cual calcula puntaje respecto a conjunto W

:return: puntaje de palabra w respecto a conjunto W
"""
def getPseudoInvertedCompactnessScore(embedding, W, w):
    sum = 0
    k = 0
    for word in W:
        if not word in embedding:
            continue

        if w in embedding:
            sum += (embedding.similarity(word, w) + embedding.similarity(w, word))

        k += 1

    return (sum / k)


"""
Elimina palabras dentro de conjunto principal y conjunto outlier, que no aparezcan en el vocabulario

:param embedding: lista de vectores de palabras
:param main_set: lista de palabras del conjunto principal
:param outlier_set: lista de palabras del conjunto outlier

:return: conjunto principal y outlier actualizado, ademas de la cantidad de palabras omitidas
"""
def omitOOVWord(embedding, main_set, outlier_set):
    res_main_set = []
    res_outlier_set = []
    main_omited = 0
    outlier_omited = 0

    for w in main_set:
        if w in embedding:
            res_main_set.append(w)
        else:
            main_omited += 1

    for w in outlier_set:
        if w in embedding:
            res_outlier_set.append(w)
        else:
            outlier_omited += 1

    return res_main_set, res_outlier_set, main_omited, outlier_omited


"""
Obtencion de puntaje del test para un solo archivo/conjunto

:param embedding: lista de vectores de palabras
:param main_set: conjunto principal de palabras
:param outlier_set: conjunto de palabras outliers
:param exist_oov: determina si se solo palabras en la interseccion de vocabularios

:return: valor de OP y OD para cada palabra en outlier_set, respecto a main_set
"""
def getFileScores(embedding, main_set, outlier_set, exist_oov):
    OP = []
    OD = []
    for outlier in outlier_set:
        # Obtencion de puntaje para outlier (nos importa su posicion respecto al resto de puntajes)
        W = main_set
        p = getPseudoInvertedCompactnessScore(embedding, W, outlier)
        pos = len(main_set)

        W_outlier = W
        for i in range(len(W_outlier)):

            C = W_outlier[0: i] + W_outlier[i+1:] + [outlier]
            w = W_outlier[i]

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
:param exist_oov: determina si pueden existir palabras fuera del vocabulario

:return: accuracy y OPP, e informacion relacionada a palabras oov
"""
def getScores(embedding, test_sets, exist_oov):
    # Suma de valores op y od
    cant_test = 0
    sum_op = 0.0
    sum_od = 0.0

    # Obtencion de porcentaje de omision
    total_main = 0
    total_main_omited = 0
    total_outlier = 0
    total_outlier_omited = 0
    total_omited_sets = 0

    count = 0
    for test in test_sets:
        count += 1
        print(">>> Test " + str(count) + " of " + str(len(test_sets)))


        # Conjunto principal y outlier
        main_set, outlier_set = test
        total_main += len(main_set)
        total_outlier += len(outlier_set)

        print(">>> Original set:", end='\n    ')
        print(main_set, end='\n    ')
        print(outlier_set)


        # Determinar cuales palabras de cada set se encuentran en el vocabulario
        if exist_oov:
            main_set, outlier_set, main_omited, outlier_omited = omitOOVWord(embedding, main_set, outlier_set)
            total_main_omited += main_omited
            total_outlier_omited += outlier_omited

            total_main += len(main_set)
            total_outlier += len(outlier_set)

            print(">>> In-vocabulary set:", end='\n    ')
            print(main_set, end='\n    ')
            print(outlier_set)


        # Omitir test si no hay palabras suficientes en el conjunto principal o en conjunto outlier
        if len(main_set) < 2 or len(outlier_set) < 1:
            total_omited_sets += 1
            print("Test set invalido, conjunto principal muy pequeño o conjunto outlier vacio\n")
            continue

        OP_list, OD_list = getFileScores(embedding, main_set, outlier_set, exist_oov)

        print("OP and OD list:")
        print(OP_list)
        print(OD_list)

        temp_sum_op = 0.0
        temp_sum_od = 0.0
        for op in OP_list:
            temp_sum_op += op

        sum_op += (temp_sum_op / len(main_set))

        for od in OD_list:
            temp_sum_od += od

        sum_od += temp_sum_od

        cant_test += len(outlier_set)

    results = []

    if cant_test == 0:
        results = ["Nan", "Nan"]
    else:
        results = [
            (sum_op / cant_test),
            (sum_od / cant_test),
            (total_main_omited * 1.0 / total_main),
            (total_outlier_omited * 1.0 / total_outlier),
            total_omited_sets
        ]

    return results


###########################################################################################
# MANEJO DE ARCHIVOS Y DATASET
###########################################################################################


"""
Obtencion de la lista de archivos test

:return: lista con nombre de archivos con test de outlier detection
"""
def getTestFiles():
    if not _DATASET.exists():
        raise Exception("No se logro encontrar carpeta con test")

    return os.listdir(_DATASET)


"""
Obtencion de las palabras desde el archivo de test, palabras del conjunto y conunto outlier

:param file_name: nombre de archivo con test
:param lower: determina si se utilizan solo minusculas en el test

:return: par de conjuntos, conjunto principal y conjunto outlier
"""
def getWords(file_name, lower):
    main_set = []
    outlier_set = []

    with io.open(_DATASET / file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if line == "\n":
                main_set = outlier_set
                outlier_set = []
                continue

            line = line.strip()
            if lower:
                line = line.lower()

            outlier_set.append(line)

    return main_set, outlier_set


"""
Metodo para la obtencion de todos los conjuntos de palabras que se utilizaran como test

:param lower: determina si se utilizan solo minusculas en el test

:return: lista de pares de conjuntos, conjunto principal y outlier, de cada test
"""
def getTests(lower):
    file_list = getTestFiles()
    test_list = []
    count = 0

    for file in file_list:
        test_list.append(getWords(file, lower))

    return test_list


###########################################################################################
# GUARDAR RESULTADOS
###########################################################################################

"""
Guardado de resultados del test por outlier detection

:param embedding_name: nombre de embedding evaluado
:param results: lista de los resultados (accuraccy, OPP) y datos (% oov outlier, % ovv main set, % grupos omitidos)
                de un embedding, la lista se compone de pares [nombre_resultado, valor_resultado]
:param exist_oov: determina si se consideran palabras fuera del vocabulario
"""
def saveResults(embedding_name, results, exist_oov):
    extension = ""

    if not exist_oov:
        extension += "_vocabIntersect"

    save_path = _RESULT
    if not save_path.exists():
        os.makedirs(save_path)

    save_path = _RESULT / (embedding_name + extension + ".txt")

    with io.open(save_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(r[0] + " " + str(r[1]) + "\n")


###########################################################################################
# EVALUACION POR OUTLIER DETECTION
###########################################################################################


"""
Realizacion de test de outlier detection

:param embedding: lista de vectores de palabras
:param embedding_name: nombre del embedding a evaluar
:param existe_ovv: determina si se consideran palabras fuera del vocabulario

:return: lista con resultados de accuraccy y OPP, ademas de info sobre palabras oov
"""
def outlierDetectionTest(embedding, embedding_name, exist_oov=True, lower=True):
    # Obtencion de conjuntos, principal y outlier
    test_list = getTests(lower)


    # Obtencion y limpieza de resultados
    results = getScores(embedding, test_list, exist_oov)
    results = [
        ["accuraccy", results[0]],
        ["OPP", results[1]],
        ["%_main_omited", results[2]],
        ["%_outlier_omited", results[3]],
        ["sets_omited", results[4]]
    ]

    print(">>> Resultados:\n    ", end='')
    print(results, end='\n\n')


    # Guardado de resultados
    saveResults(embedding_name, results, exist_oov)
    return results

