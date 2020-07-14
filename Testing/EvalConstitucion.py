from sklearn.metrics.pairwise import cosine_similarity

import ConstitucionUtil
import numpy as np

import random

###########################################################################################
# Clasificacion a partir de vectores promedio
###########################################################################################

class ConstitucionTestClass:
    def __init__(self):
        print("Constitucion test class")


    def prepareTaskA(self, gob_concept_vectors, gob_args_vectors):

        # Manejo de conceptos #

        # Inicializar diccionario, segun topico, con lista de vectores de conceptos
        gob_concept_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de conceptos
        gob_concept_concept_list_by_topics = {}

        # Inicializamos topicos
        for topic in gob_concept_vectors.keys():
            gob_concept_vectors_list_by_topics[topic] = []
            gob_concept_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto de gobierno.
        for topic in gob_concept_vectors.keys():
            print("Topico " + topic + ", cantidad de conceptos: " + str(len(gob_concept_vectors[topic].keys())))

            # Guardamos vectores y strings de conceptos
            for concept in gob_concept_vectors[topic].keys():
                concept_vector = gob_concept_vectors[topic][concept]

                if concept_vector.size == 0:
                    continue

                # Guardando vectores
                gob_concept_vectors_list_by_topics[topic].append(concept_vector)

                # Guardar concepts asociado
                gob_concept_concept_list_by_topics[topic].append(concept)

            gob_concept_vectors_list_by_topics[topic] = np.vstack(gob_concept_vectors_list_by_topics[topic])


        # Manejo de argumentos #

        # Inicializar diccionario, segun topico, con lista de vectores de argumentos
        gobc_arguments_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de argumentos
        gobc_arguments_concept_list_by_topics = {}

        # Inicializamos topicos
        for topic in gob_args_vectors.keys():
            gobc_arguments_vectors_list_by_topics[topic] = []
            gobc_arguments_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada argumento
        for topic in gob_args_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(gob_args_vectors[topic])))

            # Guardamos vectores y strings de argumentos
            for tupla in gob_args_vectors[topic]:
                concept = tupla["concept"]
                args_content = tupla["arg"]["content"]
                args_vector = tupla["arg"]["vector"]

                # Revisar que concepto abierto entregado no es nulo
                if args_content.lower() == 'null':
                    continue

                # Revisar que el concepto abierto tiene un vector promedio que lo represente
                if args_vector.size == 0:
                    continue

                gobc_arguments_vectors_list_by_topics[topic].append(args_vector)
                gobc_arguments_concept_list_by_topics[topic].append(concept)

        return [gobc_arguments_vectors_list_by_topics, gobc_arguments_concept_list_by_topics], [gob_concept_vectors_list_by_topics, gob_concept_concept_list_by_topics]


    def prepareTaskB(self, gob_concept_vectors, open_args_vectors):

        # Manejo de conceptos de gobierno #

        # Inicializar diccionario, segun topico, con lista de vectores
        gob_concept_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de conceptos
        gob_concept_concept_list_by_topics = {}

        # Inicializamos topicos
        for topic in gob_concept_vectors.keys():
            gob_concept_vectors_list_by_topics[topic] = []
            gob_concept_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto de gobierno.
        for topic in gob_concept_vectors.keys():
            print("Topico " + topic + ", cantidad de conceptos: " + str(len(gob_concept_vectors[topic].keys())))

            # Guardamos vectores y strings de conceptos
            for concept in gob_concept_vectors[topic].keys():
                concept_vector = gob_concept_vectors[topic][concept]

                if concept_vector.size == 0:
                    continue

                # Guardando vectores
                gob_concept_vectors_list_by_topics[topic].append(concept_vector)

                # Guardar concepts
                gob_concept_concept_list_by_topics[topic].append(concept)

            gob_concept_vectors_list_by_topics[topic] = np.vstack(gob_concept_vectors_list_by_topics[topic])


        # Manejo de conceptos abierto #

        # Inicializar diccionario, segun topico, con lista de vectores
        open_arguments_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de conceptos abiertos
        open_arguments_concept_list_by_topics = {}

        # Inicializamos topicos
        for topic in open_args_vectors.keys():
            open_arguments_vectors_list_by_topics[topic] = []
            open_arguments_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto abierto.
        for topic in open_args_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(open_args_vectors[topic])))

            # Guardamos vectores y strings de conceptos abiertos
            for tupla in open_args_vectors[topic]:
                equivalent_concept = tupla["concept"]
                open_concept = tupla["open_concept"]["content"]
                open_concept_vector = tupla["open_concept"]["vector"]

                # Revisar que concepto esta dentro de los conceptos de gobierno
                if not equivalent_concept in gob_concept_concept_list_by_topics[topic]:
                    continue

                # Revisar que concepto abierto entregado no es nulo
                if open_concept.lower() == 'null':
                    continue

                # Revisar que el concepto abierto tiene un vector promedio que lo represente
                if open_concept_vector.size == 0:
                    continue

                # Guardando vectores
                open_arguments_vectors_list_by_topics[topic].append(open_concept_vector)

                # Guardando concepto equivalente
                open_arguments_concept_list_by_topics[topic].append(equivalent_concept)

        return [open_arguments_vectors_list_by_topics, open_arguments_concept_list_by_topics], [gob_concept_vectors_list_by_topics, gob_concept_concept_list_by_topics]


    def prepareTaskC(self, gob_args_vectors, open_args_vectors, mode_vectors):
        # Manejo de conceptos abierto #

        # Inicializar diccionario, segun topico, con lista de vectores
        arguments_vectors_list_by_topics = {}
        # Inicializar diccionario, segun topico, con lista de conceptos abiertos
        arguments_concept_list_by_topics = {}

        # Inicializamos topicos
        for topic in open_args_vectors.keys():
            arguments_vectors_list_by_topics[topic] = []
            arguments_concept_list_by_topics[topic] = []

        # Obtenemos los vectores correspondientes a cada concepto abierto.
        for topic in open_args_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(open_args_vectors[topic])))

            # Guardamos vectores y strings de conceptos abiertos
            for tupla in (open_args_vectors[topic] + gob_args_vectors[topic]):
                argument = tupla["arg"]["content"]
                arg_vector = tupla["arg"]["vector"]
                arg_mode = tupla["mode"]

                # Revisar que argumento es de un modo valido
                if arg_mode == "blank" or arg_mode == "undefined":
                    continue

                # Revisar que argumento entregado no es nulo
                if argument.lower() == 'null':
                    continue

                # Revisar que el argumento tiene un vector promedio que lo represente
                if arg_vector.size == 0:
                    continue

                # Guardando vectores
                arguments_vectors_list_by_topics[topic].append(arg_vector)

                # Guardando concepto equivalente
                arguments_concept_list_by_topics[topic].append(arg_mode)




    def MeanVectorEvaluation(self, word_vector, word_vector_name):
        # Obtencion de datos ordenados, ademas de sus respectivos vectores promedios.
        gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors = ConstitucionUtil.getSortedDataset(
            word_vector)

        for key in gob_concept_vectors.keys():
            print(key + ": " + str(len(gob_concept_vectors[key])))

        for key in gob_args_vectors.keys():
            print(key + ": " + str(len(gob_args_vectors[key])))

        for key in open_args_vectors.keys():
            print(key + ": " + str(len(open_args_vectors[key])))

        for mode in mode_vectors:
            print(mode)

        ######################################################################################
        # Task A

        print("\nTask A")
        gobc_arguments_vec_label, gob_concept_vectors_label = self.prepareTaskA(gob_concept_vectors, gob_args_vectors)
        result_taskA = self.meanVectorClasification(gobc_arguments_vec_label[0], gobc_arguments_vec_label[1], gob_concept_vectors_label[0], gob_concept_vectors_label[1])


        ######################################################################################
        # Task B

        open_concept_vector_label, gob_concept_vectors_label = self.prepareTaskB(gob_concept_vectors, open_args_vectors)
        print("\nTask B")
        result_taskB = self.meanVectorClasification(open_concept_vector_label[0], open_concept_vector_label[1], gob_concept_vectors_label[0], gob_concept_vectors_label[1])


        ######################################################################################
        # Task C
        #arguments_vector_label, arg_mode_vectors_label = self.prepareTaskB(gob_args_vectors, open_args_vectors, mode_vectors)
        print("\nTask C")
        #result_taskC = self.meanVectorClasification(gob_args_vectors, open_args_vectors)

        # TODO: save results

        return result_taskA, result_taskB#, result_taskC


    def meanVectorClasification(self, input_vectors, input_labels, class_vector, class_label):
        acuraccy_results = {}

        # Obtencion accuracy (top1 y top5) de similaridad.
        for topic in input_vectors.keys():
            print("Topico " + topic + ": cantidad de vectores " + str(len(input_vectors[topic])))

            total_evaluado = len(input_vectors[topic])
            top5_correct = 0
            top1_correct = 0

            for i in range(len(input_vectors[topic])):
                if (i + 1) % (len(input_vectors[topic]) // 10) == 0:
                    print(" > " + str(i) + ": " + str(top1_correct) + " " + str(top5_correct))
                vector = input_vectors[topic][i]
                vector_label = input_labels[topic][i]

                # Comparando similaridad entre vectores promedios
                results = cosine_similarity(class_vector[topic], np.array([vector]))
                results = results.reshape(1, results.size)[0]

                index = np.argsort(results)
                index_most_similar = index[-1]
                index_most_similar_top5 = index[-5:]

                # Calcular si se predijo correctamente
                if vector_label == class_label[topic][index_most_similar]:
                    top1_correct += 1

                # Calcular si la prediccion es correcta en los primeros 5
                for id in index_most_similar_top5:
                    if vector_label == class_label[topic][id]:
                        top5_correct += 1
                        break


            # Calculo de accuracy para el topico
            top1_acuraccy = top1_correct / total_evaluado
            top5_acuraccy = top5_correct / total_evaluado

            print("Resultados: " + str(top1_acuraccy) + " " + str(top5_acuraccy))

            if topic not in acuraccy_results.keys():
                acuraccy_results[topic] = []

            acuraccy_results[topic] = [top1_acuraccy, top5_acuraccy]

        return acuraccy_results