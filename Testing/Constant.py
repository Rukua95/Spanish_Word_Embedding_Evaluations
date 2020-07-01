import os
from pathlib import Path

TESTING_FOLDER = Path(os.getcwd())
MAIN_FOLDER = TESTING_FOLDER.parent

EMBEDDING_FOLDER = MAIN_FOLDER / "Embeddings"
DATA_FOLDER = MAIN_FOLDER / "Datasets"
RESULTS_FOLDER = MAIN_FOLDER / "Resultados"
TEMP_RESULT_FOLDER = MAIN_FOLDER / "TempResultados"


#print(TESTING_FOLDER)
#print(MAIN_FOLDER)
#print(EMBEDDING_FOLDER)
#print(DATA_FOLDER)
#print(RESULTS_FOLDER)
#print(TEMP_RESULT_FOLDER)
