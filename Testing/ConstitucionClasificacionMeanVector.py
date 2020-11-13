from sklearn.metrics.pairwise import cosine_similarity
from ConstitucionDataHandling import getSortedDataset
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

import re
import os
import io
import Constant


###########################################################################################
# Clasificacion a partir de vectores promedio
###########################################################################################

class MeanVectorClassificationTestClass:
    total_words = []

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "_Constitucion"
    _RESULT = Constant.RESULTS_FOLDER / "Constitucion_mean_vector"
    def __init__(self, lower=True):
        print(">>> Test de Clasificacion de textos (vector promedio) <<<")

        self._lower = lower

    """
    Funcion que retorna el vector promedio de una phrase, respecto a un embedding
    """
    def meanVector(self, word_embedding, phrase):
        sum_vec = [np.zeros(word_embedding.vectorSize())]
        phrase = re.sub('[^0-9a-zA-Záéíóú]+', ' ', phrase.lower())
        phrase = phrase.strip().split()
        num = len(phrase)

        count = 0
        if num == 0:
            return np.array([])

        for word in phrase:
            if word not in word_embedding:
                count += 1

                continue

            sum_vec.append(word_embedding[word])

        if count == num:
            return np.array([])


        return (np.sum(sum_vec, axis=0) / (num - count))


    """
    Preparacion para resolver task A, entrega lista de vectores para clases y texto
    """
    def prepareTaskA(self, data, word_embedding):
        # Inicializar diccionario, segun topico, con lista de vectores de conceptos
        gob_concept_vectors_list_by_topics = {}

        # Inicializar diccionario, segun topico, con lista de nombres de conceptos
        gob_concept_concept_list_by_topics = {}

        for topic in data.keys():
            gob_concept_vectors_list_by_topics[topic] = []
            gob_concept_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto de gobierno.
        for topic in data.keys():
            print("Topico " + topic + " => cantidad de conceptos: " + str(len(data[topic].keys())))

            # Guardamos vectores y strings de conceptos
            for concept in data[topic].keys():
                concept_vector = self.meanVector(word_embedding, concept)

                if concept_vector.size == 0:
                    continue

                # Guardando vectores
                gob_concept_vectors_list_by_topics[topic].append(concept_vector)

                # Guardar concepts
                gob_concept_concept_list_by_topics[topic].append(concept)

            # Almacenamos los vectores de conceptos en una matriz
            gob_concept_vectors_list_by_topics[topic] = np.vstack(gob_concept_vectors_list_by_topics[topic])


        # Argumentos
        #############

        # Inicializar diccionario, segun topico, con lista de vectores de argumentos
        gobc_arguments_vectors_list_by_topics = {}

        # Inicializar diccionario, segun topico, con lista de concepts de argumentos
        gobc_arguments_concept_list_by_topics = {}

        for topic in data.keys():
            gobc_arguments_vectors_list_by_topics[topic] = []
            gobc_arguments_concept_list_by_topics[topic] = []

        for topic in data.keys():
            total_args = 0
            usable_args = 0
            for concept in data[topic].keys():
                print("Topico", topic," => concepto:", concept, "=> cantidad de argumentos:", len(data[topic][concept]))
                for arg in data[topic][concept]:
                    total_args += 1

                    # Revisar que concepto abierto entregado no es nulo
                    if arg.lower() == 'null':
                        continue

                    args_vector = self.meanVector(word_embedding, arg)

                    # Revisar que el concepto abierto tiene un vector promedio que lo represente
                    if args_vector.size == 0:
                        continue

                    usable_args += 1

                    gobc_arguments_vectors_list_by_topics[topic].append(args_vector)
                    gobc_arguments_concept_list_by_topics[topic].append(concept)

            print(" > Topico", topic, "=> Argumentos totales: ", total_args, "- Argumentos usables: ", usable_args)

        return [gobc_arguments_vectors_list_by_topics, gobc_arguments_concept_list_by_topics], [gob_concept_vectors_list_by_topics, gob_concept_concept_list_by_topics]


    """
    Preparacion para resolver task B
    """
    def prepareTaskB(self, data, word_embedding): # gob_concept_vectors, open_args_vectors):

        # Conceptos
        #############

        # Inicializar diccionario, segun topico, con lista de vectores de conceptos
        gob_concept_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de nombres de conceptos
        gob_concept_concept_list_by_topics = {}

        for topic in data.keys():
            gob_concept_vectors_list_by_topics[topic] = []
            gob_concept_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto de gobierno.
        for topic in data.keys():
            print("Topico " + topic + " => cantidad de conceptos: " + str(len(data[topic].keys())))

            # Guardamos vectores y strings de conceptos
            for concept in data[topic].keys():
                concept_vector = self.meanVector(word_embedding, concept)

                if concept_vector.size == 0:
                    continue

                # Guardando vectores
                gob_concept_vectors_list_by_topics[topic].append(concept_vector)

                # Guardar concepts
                gob_concept_concept_list_by_topics[topic].append(concept)

            # Almacenamos los vectores de conceptos en una matriz
            gob_concept_vectors_list_by_topics[topic] = np.vstack(gob_concept_vectors_list_by_topics[topic])


        # Conceptos abiertos
        #############

        # Inicializar diccionario, segun topico, con lista de vectores
        open_arguments_vectors_list_by_topics = {}

        # Inicializar diccionario, segun topico, con lista de vectores
        open_arguments_concept_list_by_topics = {}

        for topic in data.keys():
            open_arguments_vectors_list_by_topics[topic] = []
            open_arguments_concept_list_by_topics[topic] = []

        for topic in data.keys():
            total_concept = 0
            usable_concept = 0
            for concept in data[topic].keys():
                print("Topico", topic, " => concepto:", concept, "=> cantidad de argumentos:", len(data[topic][concept]))
                for open_concept in data[topic][concept]:
                    total_concept += 1

                    # Revisar que concepto abierto entregado no es nulo
                    if open_concept.lower() == 'null':
                        continue

                    open_concept_vector = self.meanVector(word_embedding, open_concept)

                    # Revisar que el concepto abierto tiene un vector promedio que lo represente
                    if open_concept_vector.size == 0:
                        continue

                    usable_concept += 1

                    open_arguments_vectors_list_by_topics[topic].append(open_concept_vector)
                    open_arguments_concept_list_by_topics[topic].append(concept)

            print(" > Topico", topic, "=> Conceptos abiertos totales:", total_concept, "- Conceptos abiertos usables:", usable_concept)

        return [open_arguments_vectors_list_by_topics, open_arguments_concept_list_by_topics], [gob_concept_vectors_list_by_topics, gob_concept_concept_list_by_topics]


    """
    Guardar resultados
    """
    def saveResults(self, result_taskA, result_taskB, word_vector_name):
        save_path = self._RESULT

        if not save_path.exists():
            os.makedirs(save_path)

        result_path = save_path / ("mean_vector_" + word_vector_name + ".txt")

        print(">>> Guardando resultados en:\n     " + str(result_path))
        with io.open(result_path, 'w', encoding='utf-8') as f:
            f.write("Task A results\n")
            print(result_taskA)
            for key in result_taskA.keys():
                f.write("Topico " + key + "\n")
                tupla = result_taskA[key]

                f.write("Top1 " + str(tupla[0]) + " Top5 " + str(tupla[1]) + "\n")

            f.write("Task B results\n")
            for key in result_taskB.keys():
                f.write("Topico " + key + "\n")
                tupla = result_taskB[key]

                f.write("Top1 " + str(tupla[0]) + " Top5 " + str(tupla[1]) + "\n")


    """
    Clasificacion a partir de vectores promedio
    """
    def meanVectorClasification(self, input_vectors, input_labels, class_vector, class_label):
        print(">>> Clasificacion")
        acuraccy_results = {}

        # Obtencion accuracy (top1 y top5) de similaridad.
        for topic in input_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(input_vectors[topic])))

            total_evaluado = len(input_vectors[topic])
            top5_correct = 0
            top1_correct = 0

            for i in range(len(input_vectors[topic])):
                if (i + 1) % (len(input_vectors[topic]) // 10) == 0:
                    print(" > top1_correct " + str(top1_correct) + ",top5_correct " + str(top5_correct))

                vector = input_vectors[topic][i]
                vector_label = input_labels[topic][i]

                # Comparando similaridad entre vectores promedios
                results = cosine_similarity(class_vector[topic], np.array([vector]))
                results = results.reshape(1, results.size)[0]

                index = np.argsort(results)
                index_most_similar = index[-1]
                index_most_similar_top5 = index[-5:]

                label1 = vector_label
                label2 = class_label[topic][index_most_similar]

                # Calcular si se predijo correctamente
                if label1 == label2:
                    top1_correct += 1

                # Calcular si la prediccion es correcta en los primeros 5
                for id in index_most_similar_top5:
                    if vector_label == class_label[topic][id]:
                        top5_correct += 1
                        break


            # Calculo de accuracy para el topico
            top1_acuraccy = top1_correct / total_evaluado
            top5_acuraccy = top5_correct / total_evaluado


            # Calculo de presicion y recall (Solo usado para task C)

            print(">>> Resultados => top1: ", str(top1_acuraccy), " top5: ", str(top5_acuraccy))

            if topic not in acuraccy_results.keys():
                acuraccy_results[topic] = []

            acuraccy_results[topic] = [top1_acuraccy, top5_acuraccy]

        return acuraccy_results

    """
    Evaluacion de embeddings
    """
    def MeanVectorEvaluation(self, word_vector_name, word_vector):
        print(">>> Evaluando embedding ", str(word_vector_name))

        # Obtencion de datos ordenados, ademas de sus respectivos vectores promedios.
        dataA, dataB = getSortedDataset()

        ######################################################################################
        # Task A

        print("\nTask A")
        gobc_arguments_vec_label, gob_concept_vectors_label = self.prepareTaskA(dataA, word_vector)
        result_taskA = self.meanVectorClasification(gobc_arguments_vec_label[0], gobc_arguments_vec_label[1], gob_concept_vectors_label[0], gob_concept_vectors_label[1])

        ######################################################################################
        # Task B

        print("\nTask B")
        open_concept_vector_label, gob_concept_vectors_label = self.prepareTaskB(dataB, word_vector)
        result_taskB = self.meanVectorClasification(open_concept_vector_label[0], open_concept_vector_label[1], gob_concept_vectors_label[0], gob_concept_vectors_label[1])

        # Guardamos resultados
        self.saveResults(result_taskA, result_taskB, word_vector_name)

        return result_taskA, result_taskB
