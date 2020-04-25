import os
import io
import numpy as np

from scipy.stats import spearmanr

"""
Retorna una lista de tuplas (word1, word2, score) a partir de un archivo de similaridad.
"""
def get_word_pairs(path, lower=True):
    print("Opening file: " + path)
    assert os.path.isfile(path) and type(lower) is bool
    word_pairs = []

    i = 1
    total = 0
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total = total + 1

            line = line.rstrip()
            line = line.lower() if lower else line
            line = line.split()

            # ignore phrases, only consider words
            if len(line) != 3:
                assert len(line) > 3
                continue

            pair = (line[0], line[1], float(line[2]))
            word_pairs.append(pair)

            i = i + 1

    return word_pairs


def get_spearman_rho(embeddings, path, lower):
    """
    Compute monolingual or cross-lingual word similarity score.
    """
    assert type(lower) is bool

    word_pairs = []
    for dir_path in path:
        word_pairs += get_word_pairs(dir_path)

    not_found_pairs = 0
    not_found_words = 0
    not_found_list = []

    pred = []
    gold = []
    for word1, word2, similarity in word_pairs:
        w1 = word1 in embeddings
        w2 = word2 in embeddings

        if not w1 or not w2:
            not_found_pairs += 1

            if not w1:
                not_found_words += 1
                not_found_list.append(word1)

            if not w2:
                not_found_words += 1
                not_found_list.append(word2)

            continue

        u = embeddings[word1]
        v = embeddings[word2]
        score = u.dot(v) / (np.linalg.norm(u) * np.linalg.norm(v))
        gold.append(similarity)
        pred.append(score)

    print("Not found words:" + str(not_found_words))
    for s in not_found_list:
        print("  " + s)

    return spearmanr(gold, pred).correlation, len(gold), not_found_pairs


"""
Entrega puntaje para el test de similaridad
"""
def get_wordsim_scores(dirpath, embeddings, lower=True):

    if not os.path.isdir(dirpath):
        return None

    scores = {}
    all_files = []

    # Test individuales
    print("==> Empezando test individuales")
    for filename in list(os.listdir(dirpath)):
        filepath = os.path.join(dirpath, filename)
        all_files.append(filepath)

        coeff, found, not_found = get_spearman_rho(embeddings, [filepath], lower)
        print("Not found pairs: " + str(not_found))
        scores[filename] = coeff

    # Test en conjunto
    print("==> Empezando test en conjunto")
    coeff, found, not_found = get_spearman_rho(embeddings, all_files, lower)
    scores["all_data"] = coeff

    return scores
