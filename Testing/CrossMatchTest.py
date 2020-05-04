import io
import os

import Constant

_RESULT = Constant.RESULTS_FOLDER / "CrossMatch"


def get_test_set(embedding):
    pass


def compare_test_set(sub_embedding1, sub_embedding2):
    pass

def saveResult(embedding1_name, embedding2_name, results):
    if not _RESULT.exists():
        os.makedirs(_RESULT)

    result_file = _RESULT / (embedding1_name + "_" + embedding2_name + ".txt")
    with io.open(result_file, 'w', encoding='utf-8') as f:
        f.write(str(results) + "\n")

def cross_match_test(embeddin1, embedding1_name, embedding2, embedding2_name, sample_size=1000):
    pass