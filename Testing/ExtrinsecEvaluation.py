import os
import io
import sys
import argparse

from pathlib import Path
from gensim.models.keyedvectors import KeyedVectors

from ConstitucionClasificacionMeanVector import MeanVectorClassificationTestClass
from ConstitucionClasificacionLSTM import LSTMClassificationTestClass


# Clase utilizada para representar embeddings
class Embedding:
    def __init__(self, embedding):
        self._embedding = embedding

    def __getitem__(self, item):
        return self._embedding[item]

    def __contains__(self, item):
        return (item in self._embedding)

    # Entrega tamaño de los vectores dentro del embedding
    def vectorSize(self):
        return self._embedding.vector_size

    # Entrega lista de todas las palabras en el vocabulario del embedding.
    # Las palabras estan en el mismo orden que la lista de vectores
    def getWordList(self):
        return self._embedding.index2word

    # Entrega lista de todos los vectores en el embedding.
    # Los vectores estan en el mismo orden que la lista de palabras
    def getVectors(self):
        self._embedding.init_sims()
        return self._embedding.vectors_norm

# Metodo para cargar embeddings
def getWordEmbedding(embedding_path):
    print("> Loading word embedding...")
    try:
        word_embedding = KeyedVectors.load_word2vec_format(embedding_path, limit=None)
    except:
        raise Exception("Is not posible to load the word embedding: " + str(embedding_path))
    print("  Done loading word embedding")

    return word_embedding

def multipleEmbeddingEval(args):
    # Verificar path a directorio de embeddings.
    try:
        word_embeddings_directory = Path(args.embedding_directory)
        embeddings_files_name = os.listdir(word_embeddings_directory)
    except:
        raise Exception("Problems at reading path directory with embeddings: " + str(word_embeddings_directory))

    # Obtencion de argumentos
    use_lstm = args.use_lstm_classification
    use_mean = args.use_mean_classification
    verbose = args.verbose

    for embedding_file in embeddings_files_name:
        word_embedding_path = word_embeddings_directory / embedding_file
        word_embedding_name = word_embedding_path.stem

        embedding = getWordEmbedding(word_embedding_path)
        word_embedding = Embedding(embedding)

        # Evaluacion por clasificacion de texto usando vectores promedio
        if use_mean:
            test_mean_vector = MeanVectorClassificationTestClass()
            res = test_mean_vector.MeanVectorEvaluation(word_embedding_name, word_embedding)

            print(res)

        # Evaluacion por clasificacion de texto usando red LSTM
        if use_lstm:
            test_lstm = LSTMClassificationTestClass()
            res = test_lstm.evaluate(word_embedding_name, word_embedding)

            print(res)


def singleEmbeddingEval(args):
    # Verificar path a word embedding
    try:
        word_embedding_path = Path(args.embedding_file)
        word_embedding_name = word_embedding_path.stem
    except:
        raise Exception("Problems at reading path directory with embeddings: " + str(word_embedding_path))

    embedding = getWordEmbedding(word_embedding_path)
    word_embedding = Embedding(embedding)

    use_lstm = args.use_lstm_classification
    use_mean = args.use_mean_classification
    verbose = args.verbose

    # Evaluacion por clasificacion de texto usando vectores promedio
    if use_mean:
        test_mean_vector = ConstitucionTestClass()
        res = test_mean_vector.MeanVectorEvaluation(word_embedding_name, word_embedding)

        print(res)

    # Evaluacion por clasificacion de texto usando red LSTM
    if use_lstm:
        test_lstm = RNNEvaluation()
        res = test_lstm.evaluate(word_embedding_name, word_embedding)

        print(res)


def main():

    # Parser
    my_parser = argparse.ArgumentParser(description='Word embedding extrinsec evaluation program. Word embeddings are loaded '
                                                    'using load_word2vec_format from KeyedVectors in gensim library. '
                                                    'Results are saved in folder "Resultados", under the same name as the '
                                                    'file containing the embedding. This program support evaluation by '
                                                    'text classification using mean vector or lstm for the classification '
                                                    'process.')

    my_group = my_parser.add_mutually_exclusive_group(required=True)

    # Argumentos
    my_group.add_argument('-d',
                          metavar='PATH',
                          type=str,
                          action='store',
                          dest='embedding_directory',
                          help='Path to directory containing word embeddings to evaluate.')

    my_group.add_argument('-f',
                          metavar='PATH',
                          type=str,
                          action='store',
                          dest='embedding_file',
                          help='Path to word embedding to evaluate.')

    my_parser.add_argument('-lstm',
                           action='store_true',
                           dest='use_lstm_classification',
                           help='Evaluation of word embedding by text classification, using LSTM networks for '
                                'classification.')

    my_parser.add_argument('-mean',
                           action='store_true',
                           dest='use_mean_classification',
                           help='Evaluation of word embedding by text classification, using mean vector as method of '
                                'classification.')

    my_parser.add_argument('-v',
                           '--verbose',
                           dest='verbose',
                           action='store_true',
                           help='Directly show results from evaluations and related info.')

    args = my_parser.parse_args()

    # Info from args
    word_embedding_path = args.embedding_file
    word_embeddings_directory = args.embedding_directory

    if word_embedding_path != None:
        singleEmbeddingEval(args)
    elif word_embeddings_directory != None:
        multipleEmbeddingEval(args)


if __name__ == '__main__':
    main()