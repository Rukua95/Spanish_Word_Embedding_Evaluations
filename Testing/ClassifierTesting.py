import os
import io
import json

import numpy as np

from scipy.stats import spearmanr
from gensim.models.keyedvectors import KeyedVectors

import Constant
import GlobalTest

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchtools import EarlyStopping
from torch import LongTensor
from torch.nn import Embedding, LSTM
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ConstitucionRNNEval import ClassifierModel


# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Lista con los nombres de los archivos de los embeddings
embedding_name_list = os.listdir(EMBEDDING_FOLDER)

print(">>> Embeddings a evaluar:")
for embedding in embedding_name_list:
    print("  > " + embedding)

def get_wordvector(file, cant=None):
    print("Cargando embedding " + file)
    wordvector_file = EMBEDDING_FOLDER / file
    word_vector = KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)
    print("Carga lista")
    return word_vector



word_vector = get_wordvector(embedding_name_list[0])
word_vector_name = (embedding_name_list[0]).split('.')[0]

# Clasificacion con LSTM
import ConstitucionRNNEval

RNN_test = ConstitucionRNNEval.RNNEvaluation()
RNN_test.evaluate(word_vector, word_vector_name)