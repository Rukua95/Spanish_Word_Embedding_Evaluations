import os
import io
import numpy as np

from scipy.stats import spearmanr

import Constant

_DATASET = Constant.DATA_FOLDER / "SimilarityDataset"
_RESULT = Constant.RESULTS_FOLDER / "Similarity"
_TEMP_RESULT = Constant.TEMP_RESULT_FOLDER / "Similarity"


###########################################################################################
# METRICA
###########################################################################################


"""
Obtencion de correlacion spearman rho a partir de pares de palabras y puntaje de similaridad

:param embedding: lista de vectores de palabras
:param word_pairs: lista de pares de palabras con su puntaje de similaridad

:return: lista con correlacion sperman-rho, cantidad de pares evaluados, cantidad de pares no evaluados 
         y cantidad de palabras no encontradas en el vocabulario
"""
def get_spearman_rho(embedding, word_pairs):
    not_found_pairs = 0
    not_found_words = 0
    not_found_list = []

    pred = []
    gold = []
    for word1, word2, similarity in word_pairs:
        w1 = word1 in embedding
        w2 = word2 in embedding

        if not w1 or not w2:
            not_found_pairs += 1

            if not w1:
                not_found_words += 1
                not_found_list.append(word1)

            if not w2:
                not_found_words += 1
                not_found_list.append(word2)

            continue

        u = embedding[word1]
        v = embedding[word2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)

    print("    Not found words:" + str(not_found_words))
    for word in not_found_list:
        print("    " + word)

    return spearmanr(gold, pred).correlation, len(gold), not_found_pairs, not_found_words


###########################################################################################
# MANEJO DE ARCHIVOS Y DATASET
###########################################################################################


"""
Obtencion de archivos de test desde carpeta de datasets

:return: lista de nombres de dataset
"""
def getTestFiles():
    if not _DATASET.exists():
        raise Exception("No se logro encontrar carpeta con test")

    return os.listdir(_DATASET)


"""
Obtencion de pares de palabras en algun archivo

:param file: nombre de archivo test
:param lower: determina si se cambian las mayusculas por minusculas

:return: lista con pares de palabras y su puntaje de similaridad
"""
def getWordPairs(file, lower):
    print(">>> Abriendo archivo " + file)

    word_pairs = []
    total = 0
    with io.open(_DATASET / file, 'r', encoding='utf-8') as f:
        for line in f:
            total = total + 1

            line = line.strip()
            line = line.lower() if lower else line
            line = line.split()

            # Ignorar frases y solo considerar palabras
            if len(line) != 3:
                assert len(line) > 3
                continue

            pair = (line[0], line[1], float(line[2]))
            word_pairs.append(pair)

    return word_pairs


###########################################################################################
# GUARDAR RESULTADOS
###########################################################################################


"""
Guarda resultados del test de similaridad

:param embedding_name: nombre del embedding
:param score: lista de resultados
"""
def saveResults(embedding_name, score):
    if not _RESULT.exists():
        os.makedirs(_RESULT)

    result_path = _RESULT / (embedding_name + ".txt")
    print(">>> Guardando resultados en:\n    " + str(result_path))

    with io.open(result_path, 'w', encoding='utf-8') as f:
        for tuple in score:
            for key in tuple.keys():
                f.write(key + ": " + tuple[key] + "\n")


###########################################################################################
# EVALUACION POR SIMILARITY
###########################################################################################


"""
Realizacion de test de similaridad

:return: coeficiente de correlacion spearman rho, cantidad de palabras no encontradas y cantidad de pares no evaluados
"""
def similarityTest(embedding, embedding_name, lower=True):
    test_file_list = getTestFiles()
    all_word_pairs = []
    scores = []


    # Test individuales
    print(">>> Test individuales")
    for test_file in test_file_list:
        print("    Archivo test: " + test_file)
        word_pairs = getWordPairs(test_file, lower)
        all_word_pairs = all_word_pairs + word_pairs

        coeff, found, not_found_pairs, not_found_words = get_spearman_rho(embedding, word_pairs)
        scores.append({
            test_file: coeff,
            "not_found_pairs": not_found_pairs,
            "not_found_words": not_found_words,
        })

        print("    > Cantidad de pares no procesados: " + str(not_found_pairs) + "\n\n")

    # Test en conjunto
    print(">>> Empezando test en conjunto")

    coeff, found, not_found_pairs, not_found_words = get_spearman_rho(embedding, all_word_pairs)
    scores.append({
        "all_data": coeff,
        "not_found_pairs": not_found_pairs,
        "not_found_words": not_found_words
    })

    print("    > Cantidad de pares no procesados: " + str(not_found_pairs) + "\n\n")

    # Guardando resultados
    saveResults(embedding_name, scores)

    print(">>> Resultados")
    for tuple in scores:
        print("    " + tuple[0] + ": ", end='')
        print(tuple[1:])

    print("\n")

    return scores
