import csv
import re
import GPUtil
import torch
import os
import io
import Constant

import numpy as np
import torch.nn as nn

from random import shuffle
from sklearn.metrics import precision_recall_fscore_support
from pytorchtools import EarlyStopping
from ConstitucionDataHandling import getDataTaskA, getDataTaskB
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from gensim.models.keyedvectors import KeyedVectors



# Path a carpeta principal
MAIN_FOLDER = Constant.MAIN_FOLDER

# Path a carpeta con los embeddings
EMBEDDING_FOLDER = Constant.EMBEDDING_FOLDER

# Extraccion de embeddings
def get_wordvector(file, cant=None):
    wordvector_file = EMBEDDING_FOLDER / file
    print(">>> Cargando vectores " + file + " ...", end='')
    word_vector = KeyedVectors.load_word2vec_format(wordvector_file, limit=cant)
    print("listo.\n")

    return word_vector

_DATASET = Constant.DATA_FOLDER / "_Constitucion\\constitucion_data.csv"
_RESULT = Constant.RESULTS_FOLDER / "Constitucion"


##############################################################################
# Modelo
##############################################################################

class ClassifierModel(nn.Module):
    def __init__(self, label_size, emb_weight, traing_emb=False):
        super(ClassifierModel, self).__init__()

        self.hidden_size = label_size

        if not traing_emb:
            with torch.no_grad():
                self.embedding = nn.Embedding(emb_weight.size()[0], emb_weight.size()[1])

        else:
            self.embedding = nn.Embedding(emb_weight.size()[0], emb_weight.size()[1])

        self.lstm = nn.LSTM(
            input_size=emb_weight.size()[1],
            hidden_size=label_size,
            bidirectional=False,
            batch_first=True)

        self.lstm.cuda()

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, entity_ids, lengths):
        input = self.embedding(entity_ids)
        batch_size, seq_len, feature_len = input.size()

        input = input.cuda()
        input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(input)
        output, lengths = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=False)
        output = output.cpu()

        output = output.view(batch_size * seq_len, self.hidden_size)

        adjusted_lengths = [(l - 1) * batch_size + i for i, l in enumerate(lengths)]
        lengthTensor = torch.tensor(adjusted_lengths, dtype=torch.int64)
        output = output.cpu()
        output = output.index_select(0, lengthTensor)

        logits = self.logsoftmax(output)

        return logits

##############################################################################
# Clase para dataset
##############################################################################

class ConstitucionDatasetByTopic(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data):
        self.max_lenght = max([len(x[0]) for x in data])
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][0]
        Y = self.data[idx][1]

        X = X + [0 for x in range(self.max_lenght - len(X))]

        return [X, Y]

##############################################################################
# Clase para evaluacion por RNN
##############################################################################

class RNNEvaluation():
    _embeddings_size = None
    _lower = True
    _oov_word = {}
    _batch_size = 512

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "_Constitucion"
    _RESULT = Constant.RESULTS_FOLDER / "Constitucion_rnn"
    MODEL_FOLDER = Constant.MAIN_FOLDER / "Models"

    def __init__(self, epoch=200, batch_size=512, lower=True):
        print("Test de Constitucion")

        self._epoch = epoch
        self._batch_size = batch_size
        self._lower = lower

        if not self.MODEL_FOLDER.exists():
            os.makedirs(self.MODEL_FOLDER)


    ##############################################################################
    # Ajustes en dataset
    ##############################################################################

    # Asumimos que dataset recibe texto y su clase
    def cleanDataVocab(self, data, word_vector):
        clean_data = []
        cont = 0
        cont_tot = 0

        print(" > tamaño de dataset: ", len(data))
        for pair in data:
            X = pair[0]
            Y = pair[1]
            X_new = []

            for word in X.strip().split():
                cont_tot += 1
                if word not in word_vector:
                    cont += 1
                    continue

                X_new.append(word)

            if len(X_new) == 0:
                continue

            clean_data.append([X_new, Y])

        print(" > tamaño final dataset: ", len(clean_data))
        print(" > palabras eliminadas:", cont, "de", cont_tot)
        return clean_data


    def formatData(self, data, labels, word_vector):
        new_data = []
        for pair in data:
            X = pair[0]
            X_new = []

            Y = pair[1]
            for word in X:
                X_new.append(word_vector.getWordList().index(word) + 1)
                # X_new.append(word_vector.vocab[word].index + 1)

            Y_new = labels.index(Y)

            new_data.append([X_new, Y_new])

        return new_data

    ##############################################################################
    # Preparacion para uso
    ##############################################################################

    def results(self, prediction, concept, top=5):
        try:
            predict_idx = (torch.topk(prediction, top)).indices
        except:
            predict_idx = (torch.topk(prediction, 1)).indices

        predict_idx = predict_idx.cpu()

        top1 = 0
        top5 = 0

        for i in range(len(predict_idx)):
            if concept[i] == predict_idx[i][0]:
                top1 += 1

            if concept[i] in predict_idx[i]:
                top5 += 1

        return top1, top5


    ##############################################################################
    # Training
    ##############################################################################

    def trainModel(self, model, criterion, optimizer, train_dataset, dev_dataset, save_file_name):
        train_losses = []
        avg_train_losses = []
        avg_valid_losses = []

        # Modulo para realizar early stopping
        early_stopping = EarlyStopping(patience=10, verbose=True, path=save_file_name)

        using_last_save = False
        if save_file_name.exists():
            print("Existe un modelo guardado, inicio de carga.")
            model.load_state_dict(torch.load(save_file_name))
            using_last_save = True

        print("  Entrenamiento")
        for ep in range(1, self._epoch + 1):
            print("\n  > Epoca: " + str(ep))

            data_loader_train = DataLoader(dataset=train_dataset, batch_size=self._batch_size)

            print("\nEntrenamiento - Numero de batchs:",
                  str((len(train_dataset) + self._batch_size - 1) // self._batch_size))

            GPUtil.showUtilization()

            model.train()
            model.zero_grad()
            total_top1 = 0
            total_top5 = 0
            for i, (batch, label) in enumerate(data_loader_train):
                if using_last_save:
                    print("    > Using saved model, getting results from validation")
                    using_last_save = False
                    break

                X = torch.stack(batch, dim=1)

                x = X.clone()
                x[x != 0] = 1

                seq_len = (x == 1).sum(dim=1)

                X = X.narrow(1, 0, torch.max(seq_len))

                optimizer.zero_grad()
                output = model(X.cuda(), seq_len.cuda())

                del X
                torch.cuda.empty_cache()

                loss = criterion(output.cuda(), label.cuda())
                top1, top5 = self.results(output, label)
                total_top1 += top1
                total_top5 += top5

                loss.backward()
                optimizer.step()

                del label
                del output
                torch.cuda.empty_cache()

                train_losses.append(loss.item())
                del loss
                torch.cuda.empty_cache()

            print(">>> Acc")
            print("top1:", total_top1 / len(train_dataset))
            print("top5:", total_top5 / len(train_dataset))

            train_loss = np.average(train_losses)
            valid_loss = self.validateModel(model, criterion, dev_dataset)

            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print("    > Epoca: " + str(ep))
            print("    > Train loss", train_loss)
            print("    > Valid loss", valid_loss)

            # clear lists to track next epoch
            train_losses = []
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print(">>> Early stopping")
                break

        # Cargar ultimo checkpoint del mejor modelo
        model.load_state_dict(torch.load(save_file_name))

    def validateModel(self, model, criterion, dev_dataset):
        valid_losses = []

        data_loader_dev = DataLoader(dataset=dev_dataset, batch_size=self._batch_size)

        print("\nValidacion - Numero de batchs:",
              str((len(dev_dataset) + self._batch_size - 1) // self._batch_size))
        GPUtil.showUtilization()

        model.eval()
        total_top1 = 0
        total_top5 = 0
        with torch.no_grad():
            for i, (batch, label) in enumerate(data_loader_dev):
                X = torch.stack(batch, dim=1)
                x = X.clone()
                x[x != 0] = 1
                seq_len = (x == 1).sum(dim=1)
                X = X.narrow(1, 0, torch.max(seq_len))

                output = model(X.cuda(), seq_len.cuda())

                del X
                torch.cuda.empty_cache()

                loss = criterion(output.cuda(), label.cuda())
                top1, top5 = self.results(output, label)
                total_top1 += top1
                total_top5 += top5

                del label
                del output
                torch.cuda.empty_cache()

                valid_losses.append(loss.item())
                del loss
                torch.cuda.empty_cache()

            print(">>> Acc")
            print("top1:", total_top1 / len(dev_dataset))
            print("top5:", total_top5 / len(dev_dataset))

            return np.average(valid_losses)


    def testModel(self, model, test_dataset, label_list):
        data_loader_test = DataLoader(dataset=test_dataset, batch_size=self._batch_size)

        print("\nTest - Numero de batchs:",
              str((len(test_dataset) + self._batch_size - 1) // self._batch_size))
        GPUtil.showUtilization()

        model.eval()
        total_top1 = 0
        total_top5 = 0
        list_of_predictions = []
        list_of_labels = []
        with torch.no_grad():
            for i, (batch, label) in enumerate(data_loader_test):
                X = torch.stack(batch, dim=1)
                x = X.clone()
                x[x != 0] = 1
                seq_len = (x == 1).sum(dim=1)
                X = X.narrow(1, 0, torch.max(seq_len))

                output = model(X.cuda(), seq_len.cuda())

                del X
                torch.cuda.empty_cache()

                top1, top5 = self.results(output, label)
                total_top1 += top1
                total_top5 += top5

                list_of_predictions.append(output.clone())
                list_of_labels.append(label.clone())

                del label
                del output
                torch.cuda.empty_cache()

            prediction = torch.cat(list_of_predictions)
            prediction = ((torch.topk(prediction, 1)).indices).numpy()
            label = torch.cat(list_of_labels).numpy()
            res = precision_recall_fscore_support(label, prediction, labels=np.array(range(len(label_list))),
                                                  average='macro')

            # Calculo de presicion, recall y F1
            print(">>> Acc")
            print("Top1:", total_top1 / len(test_dataset))
            print("Top5:", total_top5 / len(test_dataset))

            return [total_top1 / len(test_dataset), total_top5 / len(test_dataset), res[0], res[1], res[2]]

    def trainAndTestTaskA(self, word_vector, word_vector_name):
        # Preparacion
        print(">>> Entrenamiento & test para task A y B")
        GPUtil.showUtilization()
        resultsA = {}
        resultsB = {}


        # Obtencion de datos para task A y task B
        print("> Obtencion de datos")
        train, dev, test = getDataTaskA()
        data_task_B = getDataTaskB()

        for topic in train.keys():
            save_path = self._RESULT
            result_name_taskA = "TaskA_topic" + str(topic)
            result_name_taskB_concat = "TaskB_concat_topic" + str(topic)
            result_name_taskB_concep = "TaskB_concep_topic" + str(topic)

            result_path = save_path / ("rnn_" + word_vector_name)
            result_file_name = result_path / (result_name_taskA + ".txt")

            if result_file_name.exists():
                result_path_Bconcat = result_path / (result_name_taskB_concat + ".txt")
                result_path_Bconcep = result_path / (result_name_taskB_concep + ".txt")

                if result_path_Bconcat.exists() and result_path_Bconcep.exists():
                    print("Para topico", topic, "ya existen resultados para task A y B")
                    continue
                else:
                    print("Aun faltan resultados de task B en topico", topic)

            print("\n> Entrenando topico", topic)
            GPUtil.showUtilization()

            # Separacion de datos de A
            data_train_topic = train[topic]
            data_dev_topic = dev[topic]
            data_test_topic = test[topic]

            data_task_B_topic = data_task_B[topic]

            print("\nEjemplos de datos")
            for data in data_train_topic[:5]:
                print(data)


            # Limpieza de datos
            clean_train = self.cleanDataVocab(data_train_topic, word_vector)
            clean_dev = self.cleanDataVocab(data_dev_topic, word_vector)
            clean_test = self.cleanDataVocab(data_test_topic, word_vector)

            print("\nEjemplos de data limpio")
            for data in clean_train[:5]:
                print(data)

            # Obtener distintos labels y su cantidad de repeticiones
            concept_list = []
            amount = {}
            for pair in clean_train:
                concept = pair[1]
                if concept not in concept_list:
                    concept_list.append(concept)

                    amount[concept] = 0
                amount[concept] += 1


            # Pasar datos a formato numerico
            formated_train = self.formatData(clean_train, concept_list, word_vector)
            formated_dev = self.formatData(clean_dev, concept_list, word_vector)
            formated_test = self.formatData(clean_test, concept_list, word_vector)

            print("\nCantida de datos finales:", len(formated_train))
            print("Ejemplos de datos formateados")
            for data in formated_train[:5]:
                print(data)

            print("\nCantidad de clases:", len(concept_list))
            print("Informacion sobre clases: indice, nombre, cantidad de datos")
            for c in concept_list:
                print(concept_list.index(c), c, amount[c])


            # Cantidad de datos por clase
            count_by_concept = [amount[c] for c in concept_list]
            weight_loss = 1. / torch.tensor(count_by_concept, dtype=torch.float32)


            # Dataset
            train_dataset = ConstitucionDatasetByTopic(formated_train)
            dev_dataset = ConstitucionDatasetByTopic(formated_dev)
            test_dataset = ConstitucionDatasetByTopic(formated_test)


            # Agregar vector de pad en embeddings
            # weight = torch.FloatTensor(word_vector.vectors)
            # pad = torch.FloatTensor([np.random.rand(word_vector.vector_size)])

            weight = torch.FloatTensor(word_vector.getVectors())
            pad = torch.FloatTensor([np.random.rand(word_vector.vectorSize())])
            weight = torch.cat([pad, weight])

            n_output = len(concept_list)


            # Definicion de modelo RNN
            mylstm = ClassifierModel(n_output, weight, traing_emb=True).cuda()
            print("\nInformacion del modelo")
            print(mylstm)

            # Definicion de optimizador
            parameters = filter(lambda p: p.requires_grad, mylstm.parameters())
            print(type(parameters))
            optimizer = torch.optim.Adam(parameters, lr=0.001)

            # Funcion de perdida para modelo
            criterion = nn.NLLLoss(weight=weight_loss.cuda())

            # Path para guardar modelo
            early_save = self.MODEL_FOLDER / (
                    "taskA_" + str(self._embeddings_size) + "_" + str(topic) + "_" + word_vector_name + ".pt")
            print("Path guardado de modelo:", str(early_save))

            self.trainModel(mylstm, criterion, optimizer, train_dataset, dev_dataset, early_save)


            # Testing para task A
            print("\n> Evaluando modelo para task A")
            t1, t5, p, r, f1 = self.testModel(mylstm, test_dataset, concept_list)
            resultsA[topic] = [t1, t5]
            result = {"Top1": t1, "Top5": t5}
            self.saveResults(word_vector_name, result, result_name_taskA)
            torch.cuda.empty_cache()


            # Caso de argumento + concepto
            print("\n> Evaluando modelo para task B: argumento + concepto")
            concat_B_task_data = [[d[0] + " " + d[1], d[2]] for d in data_task_B_topic]

            print("  Ejemplo")
            for d in concat_B_task_data[:5]:
                print(d)

            clean_task_B_data = self.cleanDataVocab(concat_B_task_data, word_vector)
            format_task_B_data = self.formatData(clean_task_B_data, concept_list, word_vector)
            task_B_data = ConstitucionDatasetByTopic(format_task_B_data)

            t1, t5, p, r, f1 = self.testModel(mylstm, task_B_data, concept_list)
            resultsB[topic] = [t1, t5]
            result = {"Top1": t1, "Top5": t5}
            self.saveResults(word_vector_name, result, result_name_taskB_concat)


            # Caso de solo concepto
            print("\n> Evaluando modelo para task B: concepto")
            concep_B_task_data = [[d[1], d[2]] for d in data_task_B_topic]

            print("  Ejemplo")
            for d in concep_B_task_data[:5]:
                print(d)

            clean_task_B_data = self.cleanDataVocab(concep_B_task_data, word_vector)
            format_task_B_data = self.formatData(clean_task_B_data, concept_list, word_vector)
            task_B_data = ConstitucionDatasetByTopic(format_task_B_data)

            t1, t5, p, r, f1 = self.testModel(mylstm, task_B_data, concept_list)
            resultsB[topic] = resultsB[topic] + [t1, t5]
            result = {"Top1": t1, "Top5": t5}
            self.saveResults(word_vector_name, result, result_name_taskB_concep)

            del mylstm
            torch.cuda.empty_cache()


    def saveResults(self, word_vector_name, result, result_name):
        print("Guardando resultados")
        save_path = self._RESULT
        result_path = save_path / ("rnn_" + word_vector_name)

        if not result_path.exists():
            os.makedirs(result_path)

        result_file_name = result_path / (result_name + ".txt")
        print("Path:", str(result_file_name))
        with io.open(result_file_name, 'w') as f:
            for key in result.keys():
                f.write(key + "\t" + str(result[key]) + "\n")


    def evaluate(self, word_vector, word_vector_name):
        self._embeddings_size = len(word_vector.getWordList())
        print("Embedding size:", self._embeddings_size)

        # Task A y B
        self.trainAndTestTaskA(word_vector, word_vector_name)






"""
class RNNEvaluation():
    _embeddings_name_list = os.listdir(EMBEDDING_FOLDER)
    _embeddings_size = None
    _lower = True
    _oov_word = {}
    _batch_size = 512

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "_Constitucion"
    _RESULT = Constant.RESULTS_FOLDER / "Constitucion_rnn"
    MODEL_FOLDER = Constant.MAIN_FOLDER / "Models"


    def __init__(self, epoch=200, batch_size=512, lower=True):
        print("Test de Constitucion")

        self._epoch = epoch
        self._batch_size = batch_size
        self._lower = lower

        if not self.MODEL_FOLDER.exists():
            os.makedirs(self.MODEL_FOLDER)

    ##############################################################################
    # Ajustes en dataset
    ##############################################################################

    # Asumimos que dataset recibe texto y su clase
    def cleanDataVocab(self, data, word_vector):
        clean_data = []
        for pair in data:
            X = pair[0]
            X_new = []
            Y = pair[1]

            for word in X.strip().split():
                if word not in word_vector:
                    X_new.append("<unk>")
                    continue

                X_new.append(word)

            if len(X_new) == 0:
                continue

            clean_data.append([X_new, Y])

        return clean_data


    def formatData(self, data, labels, word_vector):
        new_data = []
        for pair in data:
            X = pair[0]
            X_new = []
            Y = pair[1]
            Y_new = 0

            for word in X:
                if word == "<unk>":
                    X_new.append(1)
                    continue

                X_new.append(word_vector.vocab[word].index + 2)

            Y_new = labels.index(Y)

            new_data.append([X_new, Y_new])

        return new_data

    ##############################################################################
    # Preparacion para uso
    ##############################################################################

    def results(self, prediction, concept, top=5):
        try:
            predict_idx = (torch.topk(prediction, top)).indices
        except:
            predict_idx = (torch.topk(prediction, 1)).indices

        predict_idx = predict_idx.cpu()

        top1 = 0
        top5 = 0

        for i in range(len(predict_idx)):
            if concept[i] == predict_idx[i][0]:
                top1 += 1

            if concept[i] in predict_idx[i]:
                top5 += 1

        return top1, top5


    ##############################################################################
    # Training
    ##############################################################################

    def trainModel(self, model, criterion, optimizer, train_dataset, dev_dataset, save_file_name):
        train_losses = []
        avg_train_losses = []
        avg_valid_losses = []

        # Modulo para realizar early stopping
        early_stopping = EarlyStopping(patience=10, verbose=True, path=save_file_name)

        using_last_save = False
        if save_file_name.exists():
            print("Existe un modelo guardado, inicio de carga.")
            model.load_state_dict(torch.load(save_file_name))
            using_last_save = True

        print("  Entrenamiento")
        for ep in range(1, self._epoch + 1):
            print("\n  > Epoca: " + str(ep))

            data_loader_train = DataLoader(dataset=train_dataset, batch_size=self._batch_size)

            print("\nEntrenamiento - Numero de batchs:",
                  str((len(train_dataset) + self._batch_size - 1) // self._batch_size))

            GPUtil.showUtilization()

            model.train()
            model.zero_grad()
            total_top1 = 0
            total_top5 = 0
            for i, (batch, label) in enumerate(data_loader_train):
                if using_last_save:
                    print("    > Using saved model, getting results from validation")
                    using_last_save = False
                    break

                X = torch.stack(batch, dim=1)

                x = X.clone()
                x[x != 0] = 1

                seq_len = (x == 1).sum(dim=1)

                X = X.narrow(1, 0, torch.max(seq_len))

                optimizer.zero_grad()
                output = model(X, seq_len)

                del X
                torch.cuda.empty_cache()

                loss = criterion(output.cpu(), label.cpu())
                top1, top5 = self.results(output, label)
                total_top1 += top1
                total_top5 += top5

                loss.backward()
                optimizer.step()

                del label
                del output
                torch.cuda.empty_cache()

                train_losses.append(loss.item())
                del loss
                torch.cuda.empty_cache()

            print(">>> Acc")
            print("top1:", total_top1 / len(train_dataset))
            print("top5:", total_top5 / len(train_dataset))

            train_loss = np.average(train_losses)
            valid_loss = self.validateModel(model, criterion, dev_dataset)

            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)

            print("    > Epoca: " + str(ep))
            print("    > Train loss", train_loss)
            print("    > Valid loss", valid_loss)

            # clear lists to track next epoch
            train_losses = []
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print(">>> Early stopping")
                break

        # Cargar ultimo checkpoint del mejor modelo
        model.load_state_dict(torch.load(save_file_name))

    def validateModel(self, model, criterion, dev_dataset):
        valid_losses = []

        data_loader_dev = DataLoader(dataset=dev_dataset, batch_size=self._batch_size)

        print("\nValidacion - Numero de batchs:",
              str((len(dev_dataset) + self._batch_size - 1) // self._batch_size))
        GPUtil.showUtilization()

        model.eval()
        total_top1 = 0
        total_top5 = 0
        with torch.no_grad():
            for i, (batch, label) in enumerate(data_loader_dev):
                X = torch.stack(batch, dim=1)
                x = X.clone()
                x[x != 0] = 1
                seq_len = (x == 1).sum(dim=1)
                X = X.narrow(1, 0, torch.max(seq_len))

                output = model(X, seq_len)

                del X
                torch.cuda.empty_cache()

                loss = criterion(output.cpu(), label.cpu())
                top1, top5 = self.results(output, label)
                total_top1 += top1
                total_top5 += top5

                del label
                del output
                torch.cuda.empty_cache()

                valid_losses.append(loss.item())
                del loss
                torch.cuda.empty_cache()

            print(">>> Acc")
            print("top1:", total_top1 / len(dev_dataset))
            print("top5:", total_top5 / len(dev_dataset))

            return np.average(valid_losses)


    def testModel(self, model, test_dataset, label_list):
        data_loader_test = DataLoader(dataset=test_dataset, batch_size=self._batch_size)

        print("\nTest - Numero de batchs:",
              str((len(test_dataset) + self._batch_size - 1) // self._batch_size))
        GPUtil.showUtilization()

        model.eval()
        total_top1 = 0
        total_top5 = 0
        list_of_predictions = []
        list_of_labels = []
        with torch.no_grad():
            for i, (batch, label) in enumerate(data_loader_test):
                X = torch.stack(batch, dim=1)
                x = X.clone()
                x[x != 0] = 1
                seq_len = (x == 1).sum(dim=1)
                X = X.narrow(1, 0, torch.max(seq_len))

                output = model(X, seq_len)

                del X
                torch.cuda.empty_cache()

                top1, top5 = self.results(output, label)
                total_top1 += top1
                total_top5 += top5

                list_of_predictions.append(output.clone())
                list_of_labels.append(label.clone())

                del label
                del output
                torch.cuda.empty_cache()

            prediction = torch.cat(list_of_predictions)
            prediction = ((torch.topk(prediction, 1)).indices).numpy()
            label = torch.cat(list_of_labels).numpy()
            res = precision_recall_fscore_support(label, prediction, labels=np.array(range(len(label_list))),
                                                  average='macro')

            # Calculo de presicion, recall y F1
            print(">>> Acc")
            print("Top1:", total_top1 / len(test_dataset))
            print("Top5:", total_top5 / len(test_dataset))

            return [total_top1 / len(test_dataset), total_top5 / len(test_dataset), res[0], res[1], res[2]]

    def trainAndTestTaskAandB(self, word_vector, word_vector_name):
        # Preparacion
        print(">>> Entrenamiento & test para task A y B")
        GPUtil.showUtilization()
        resultsA = {}
        resultsB = {}


        # Obtencion de datos para task A y task B
        print("> Obtencion de datos")
        train, dev, test = getDataTaskA()
        data_task_B = getDataTaskB()

        for topic in train.keys():
            save_path = self._RESULT
            result_name_taskA = "TaskA_topic" + str(topic)
            result_name_taskB_concat = "TaskB_concat_topic" + str(topic)
            result_name_taskB_concep = "TaskB_concep_topic" + str(topic)

            result_path = save_path / ("rnn_" + word_vector_name)
            result_file_name = result_path / (result_name_taskA + ".txt")

            if result_file_name.exists():
                result_path_Bconcat = result_path / (result_name_taskB_concat + ".txt")
                result_path_Bconcep = result_path / (result_name_taskB_concep + ".txt")

                if result_path_Bconcat.exists() and result_path_Bconcep.exists():
                    print("Para topico", topic, "ya existen resultados para task A y B")
                    continue
                else:
                    print("Aun faltan resultados de task B en topico", topic)

            print("\n> Entrenando topico", topic)
            GPUtil.showUtilization()

            # Separacion de datos de A
            data_train_topic = train[topic]
            data_dev_topic = dev[topic]
            data_test_topic = test[topic]

            data_task_B_topic = data_task_B[topic]

            print("\nEjemplos de datos")
            for data in data_train_topic[:5]:
                print(data)


            # Limpieza de datos
            clean_train = self.cleanDataVocab(data_train_topic, word_vector)
            clean_dev = self.cleanDataVocab(data_dev_topic, word_vector)
            clean_test = self.cleanDataVocab(data_test_topic, word_vector)

            print("\nEjemplos de data limpio")
            for data in clean_train[:5]:
                print(data)

            # Obtener distintos labels y su cantidad de repeticiones
            concept_list = []
            amount = {}
            for pair in clean_train:
                concept = pair[1]
                if concept not in concept_list:
                    concept_list.append(concept)

                    amount[concept] = 0
                amount[concept] += 1


            # Pasar datos a formato numerico
            formated_train = self.formatData(clean_train, concept_list, word_vector)
            formated_dev = self.formatData(clean_dev, concept_list, word_vector)
            formated_test = self.formatData(clean_test, concept_list, word_vector)

            print("\nCantida de datos finales:", len(formated_train))
            print("Ejemplos de datos formateados")
            for data in formated_train[:5]:
                print(data)

            print("\nCantidad de clases:", len(concept_list))
            print("Informacion sobre clases: indice, nombre, cantidad de datos")
            for c in concept_list:
                print(concept_list.index(c), c, amount[c])


            # Cantidad de datos por clase
            count_by_concept = [amount[c] for c in concept_list]
            weight_loss = 1. / torch.tensor(count_by_concept, dtype=torch.float32)


            # Dataset
            train_dataset = ConstitucionDatasetByTopic(formated_train)
            dev_dataset = ConstitucionDatasetByTopic(formated_dev)
            test_dataset = ConstitucionDatasetByTopic(formated_test)


            # Agregar vector de pad en embeddings
            weight = torch.FloatTensor(word_vector.vectors)
            weight = torch.cat([self._pad_vector, weight])

            n_output = len(concept_list)


            # Definicion de modelo RNN
            mylstm = ClassifierModel(n_output, weight)
            print("\nInformacion del modelo")
            print(mylstm)

            # Definicion de optimizador
            parameters = filter(lambda p: p.requires_grad, mylstm.parameters())
            optimizer = torch.optim.Adam(parameters, lr=0.001)

            # Funcion de perdida para modelo
            criterion = nn.NLLLoss(weight=weight_loss)

            # Path para guardar modelo
            early_save = self.MODEL_FOLDER / (
                    "taskA_" + str(self._embeddings_size) + "_" + str(topic) + "_" + word_vector_name + ".pt")
            print("Path guardado de modelo:", str(early_save))

            self.trainModel(mylstm, criterion, optimizer, train_dataset, dev_dataset, early_save)


            # Testing para task A
            print("\n> Evaluando modelo para task A")
            t1, t5, p, r, f1 = self.testModel(mylstm, test_dataset, concept_list)
            resultsA[topic] = [t1, t5]
            result = {"Top1": t1, "Top5": t5}
            self.saveResults(word_vector_name, result, result_name_taskA)
            torch.cuda.empty_cache()


            # Caso de argumento + concepto
            print("\n> Evaluando modelo para task B: argumento + concepto")
            concat_B_task_data = [[d[0] + " " + d[1], d[2]] for d in data_task_B_topic]

            print("  Ejemplo")
            for d in concat_B_task_data[:5]:
                print(d)

            clean_task_B_data = self.cleanDataVocab(concat_B_task_data, word_vector)
            format_task_B_data = self.formatData(clean_task_B_data, concept_list, word_vector)
            task_B_data = ConstitucionDatasetByTopic(format_task_B_data)

            t1, t5, p, r, f1 = self.testModel(mylstm, task_B_data, concept_list)
            resultsB[topic] = [t1, t5]
            result = {"Top1": t1, "Top5": t5}
            self.saveResults(word_vector_name, result, result_name_taskB_concat)


            # Caso de solo concepto
            print("\n> Evaluando modelo para task B: concepto")
            concep_B_task_data = [[d[1], d[2]] for d in data_task_B_topic]

            print("  Ejemplo")
            for d in concep_B_task_data[:5]:
                print(d)

            clean_task_B_data = self.cleanDataVocab(concep_B_task_data, word_vector)
            format_task_B_data = self.formatData(clean_task_B_data, concept_list, word_vector)
            task_B_data = ConstitucionDatasetByTopic(format_task_B_data)

            t1, t5, p, r, f1 = self.testModel(mylstm, task_B_data, concept_list)
            resultsB[topic] = resultsB[topic] + [t1, t5]
            result = {"Top1": t1, "Top5": t5}
            self.saveResults(word_vector_name, result, result_name_taskB_concep)


            del mylstm
            torch.cuda.empty_cache()


    def saveResults(self, word_vector_name, result, result_name):
        print("Guardando resultados")
        save_path = self._RESULT
        result_path = save_path / ("rnn_" + word_vector_name)

        if not result_path.exists():
            os.makedirs(result_path)

        result_file_name = result_path / (result_name + ".txt")
        print("Path:", str(result_file_name))
        with io.open(result_file_name, 'w') as f:
            for key in result.keys():
                f.write(key + "\t" + str(result[key]) + "\n")


    def evaluate(self, word_vector, word_vector_name):
        self._embeddings_size = len(word_vector.vocab)
        print("Embedding size:", self._embeddings_size)

        self._pad_vector = torch.FloatTensor([np.random.rand(word_vector.vector_size)])

        # Task A y B
        self.trainAndTestTaskAandB(word_vector, word_vector_name)

"""