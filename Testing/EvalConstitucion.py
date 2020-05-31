from sklearn.metrics.pairwise import cosine_similarity

import ConstitucionUtil
import numpy as np

import random

###########################################################################################
# Clasificacion a partir de vectores promedio
###########################################################################################


def MeanVectorEvaluation(word_vector, word_vector_name):
    # Obtencion de datos ordenados, ademas de sus respectivos vectores promedios.
    gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors = ConstitucionUtil.getSortedDataset(
        word_vector)

    # Task A
    print("\nTask A")
    result_taskA = meanVectorEvalTaskA(gob_args_vectors)

    # Task B
    print("\nTask B")
    result_taskB = meanVectorEvalTaskB(gob_concept_vectors, open_args_vectors)

    # Task C
    print("\nTask C")
    result_taskC = meanVectorEvalTaskC(gob_args_vectors, open_args_vectors)

    return [result_taskA, result_taskB, result_taskC]


def meanVectorEvalTaskA(gob_args_vectors):
    total = 0
    total_evaluado = 0
    args_by_concept_by_topic = {}

    # Organizar argumentos por topico y por concepto
    for topic in gob_args_vectors.keys():
        print("Topico: " + topic)

        if topic not in args_by_concept_by_topic.keys():
            args_by_concept_by_topic[topic] = {}

        total += len(gob_args_vectors[topic])

        # Organizamos vectores segun concept
        for tupla in gob_args_vectors[topic]:
            concept = tupla["concept"]
            arg_vector = tupla["arg"]["vector"]

            # Omitimos vectores vacios
            if arg_vector.size == 0:
                continue

            total_evaluado += 1

            if concept not in args_by_concept_by_topic[topic].keys():
                args_by_concept_by_topic[topic][concept] = []

            args_by_concept_by_topic[topic][concept].append(arg_vector)

        for concept in args_by_concept_by_topic[topic].keys():
            print(" > " + concept)

    print("total de argumentos: " + str(total))
    print("total a evaluar: " + str(total_evaluado))

    # Experimentos
    repetitions = 5
    final_result_top1 = {}
    final_result_top5 = {}
    for h in range(repetitions):
        print("Experimento " + str(h + 1))

        # Separamos en train y test set
        for topic in args_by_concept_by_topic.keys():
            print(" > Topico " + topic)

            test_set = {}
            model = []
            concept_list = []

            # Separamos set y organizamos segun concepto
            for concept in args_by_concept_by_topic[topic].keys():
                total_args = len(args_by_concept_by_topic[topic][concept])
                print("    > " + concept + ", cantidad de args: " + str(total_args))
                print("      train_set size = " + str(int(total_args * 0.8)) + ", test_set size = " + str(
                    total_args - int(total_args * 0.8)))

                idx = random.sample(range(total_args), int(total_args * 0.8))

                train_set = []

                for i in range(total_args):
                    if i in idx:
                        train_set.append(args_by_concept_by_topic[topic][concept][i])
                    else:
                        # Test set esta separado por concepto
                        if concept not in test_set.keys():
                            test_set[concept] = []

                        test_set[concept].append(args_by_concept_by_topic[topic][concept][i])

                # Entrenamos el modelo
                train_set = np.vstack(train_set)
                model.append(np.mean(train_set, axis=0))

                concept_list.append(concept)

            # Realizamos test
            total_test_size = 0
            top1_correct = 0
            top5_correct = 0
            for concept in test_set.keys():
                total_test_size += len(test_set[concept])

                for arg_vec in test_set[concept]:
                    results = cosine_similarity(model, np.array([arg_vec]))
                    results = results.reshape(1, results.size)[0]

                    index = np.argsort(results)
                    index_most_similar = index[-1]
                    index_most_similar_top5 = index[-5:]

                    # Calcular si la prediccion es correcta
                    if concept == concept_list[index_most_similar]:
                        top1_correct += 1

                    # Calcular si la prediccion es correcta en los primeros 5
                    for id in index_most_similar_top5:
                        if concept == concept_list[id]:
                            top5_correct += 1
                            break

            print(" > topic " + topic + " results")
            print(top1_correct / total_test_size)
            print(top5_correct / total_test_size)

            if topic not in final_result_top1.keys():
                final_result_top1[topic] = 0
                final_result_top5[topic] = 0

            final_result_top1[topic] += top1_correct / total_test_size
            final_result_top5[topic] += top5_correct / total_test_size

    for topic in final_result_top1.keys():
        final_result_top1[topic] = final_result_top1[topic] / repetitions
        final_result_top5[topic] = final_result_top5[topic] / repetitions

    print("Final results: ", end='')
    print([final_result_top1, final_result_top5])

    return [final_result_top1, final_result_top5]


def meanVectorEvalTaskB(gob_concept_vectors, open_args_vectors):
    ######################################################################################
    # Organizacion de datos
    ######################################################################################

    # Diccionario, segun topico, con lista de vectores
    gob_concept_vectors_list_by_topics = {}

    # Diccionario, segun topico, con lista de conceptos
    gob_concept_list_by_topics = {}

    # Obtenemos los vectores correspondientes a cada concepto de gobierno.
    for topic in gob_concept_vectors.keys():
        # Inicialicion de diccionario para guardar vectores correspondientes a conceptos de gobierno
        if not topic in gob_concept_vectors_list_by_topics.keys():
            gob_concept_vectors_list_by_topics[topic] = np.array([])
            gob_concept_list_by_topics[topic] = []

        print("Topico " + topic + ", cantidad de conceptos: " + str(len(gob_concept_vectors[topic].keys())))

        # Guardamos vectores y strings de conceptos
        for concept in gob_concept_vectors[topic].keys():
            if gob_concept_vectors[topic][concept].size == 0:
                continue

            mean_vector = gob_concept_vectors[topic][concept]

            # Guardar vectores
            if gob_concept_vectors_list_by_topics[topic].size == 0:
                gob_concept_vectors_list_by_topics[topic] = mean_vector
            else:
                gob_concept_vectors_list_by_topics[topic] = np.vstack(
                    (gob_concept_vectors_list_by_topics[topic], mean_vector))

            # Guardar concepts
            gob_concept_list_by_topics[topic].append(concept)

    for topic in gob_concept_list_by_topics.keys():
        for concept in gob_concept_list_by_topics[topic]:
            print(" > T: " + topic + " C: " + concept)

    ######################################################################################

    acuraccy_results = []

    # Cantidad de argumentos por topico
    total_open_args = 0
    for topic in open_args_vectors.keys():
        print(topic + " " + str(len(open_args_vectors[topic])))
        total_open_args += len(open_args_vectors[topic])

    print("total: " + str(total_open_args) + "\n")

    # Obtencion accuracy (top1 y top5) de similaridad.
    for topic in open_args_vectors.keys():
        print("Topico " + topic + ": cantidad de vectores " + str(len(open_args_vectors[topic])))

        total = 0
        total_evaluado = 0
        top5_correct = 0
        top1_correct = 0

        for tupla in open_args_vectors[topic]:
            equivalent_concept = tupla["concept"]
            open_concept = tupla["open_concept"]["content"]
            open_concept_vector = tupla["open_concept"]["vector"]

            # Revisar que concepto esta dentro de los conceptos de gobierno
            if not equivalent_concept in gob_concept_list_by_topics[topic]:
                continue

            total += 1

            # Revisar que concepto abierto entregado no es nulo
            if open_concept.lower() == 'null':
                continue

            # Revisar que el concepto abierto tiene un vector promedio que lo represente
            if open_concept_vector.size == 0:
                continue

            total_evaluado += 1

            # Comparando similaridad entre vectores promedios
            results = cosine_similarity(gob_concept_vectors_list_by_topics[topic], np.array([open_concept_vector]))
            results = results.reshape(1, results.size)[0]

            index = np.argsort(results)
            index_most_similar = index[-1]
            index_most_similar_top5 = index[-5:]

            # Calcular si se predijo correctamente
            if equivalent_concept == gob_concept_list_by_topics[topic][index_most_similar]:
                top1_correct += 1

            # Calcular si la prediccion es correcta en los primeros 5
            for id in index_most_similar_top5:
                if equivalent_concept == gob_concept_list_by_topics[topic][id]:
                    top5_correct += 1
                    break

        # Calculo de accuracy para el topico
        top1_acuraccy = top1_correct / total_evaluado
        top5_acuraccy = top5_correct / total_evaluado

        print("Resultados: " + str(top1_acuraccy) + " " + str(top5_acuraccy))

        acuraccy_results.append([top1_acuraccy, top5_acuraccy, total_evaluado, total])

    total_final = 0
    for r in acuraccy_results:
        total_final += r[-1]

        print("Datos: ", end='')
        print(r)
        print("% omitidos: " + str(1 - r[-2] / r[-1]))

    print("total final: " + str(total_final))

    return acuraccy_results


def meanVectorEvalTaskC(gob_args_vectors, open_args_vectors):
    total_argumentos = 0
    total_evaluado = 0

    # Separamos los argumentos segun modo de argumentacion
    arguments_by_mode = {}
    for key in gob_args_vectors.keys():
        total_argumentos += len(gob_args_vectors[key] + open_args_vectors[key])

        for tupla in (gob_args_vectors[key] + open_args_vectors[key]):
            arg_vec = tupla["arg"]["vector"]
            mode = tupla["mode"]

            # Eliminamos argumentos que no tengan vectores
            if arg_vec.size == 0:
                continue

            # Eliminamos argumentos que sean indefinido/indeterminables
            if mode == "undefined" or mode == "blank":
                continue

            total_evaluado += 1

            if mode not in arguments_by_mode.keys():
                arguments_by_mode[mode] = []

            # Guardamos vector de argumento segun modo argumentativo
            arguments_by_mode[mode].append(arg_vec)

    print("Total de argumentos: " + str(total_argumentos))
    print("Total a evaluar: " + str(total_evaluado))

    for mode in arguments_by_mode.keys():
        print(mode + ", cantidad de argumentos: " + str(len(arguments_by_mode[mode])))

    # Realizamos los experimentos
    final_accuracy = 0
    repetitions = 5
    for h in range(repetitions):
        print("Experiment " + str(h + 1))
        test_set = {}
        mode_mean_vectors = []
        modes = []

        # Usamos train set para obtener vector promedio para cada modo de argumentacion
        for key in arguments_by_mode.keys():
            modes.append(key)

            if key not in test_set.keys():
                test_set[key] = []

            # Realizamos la separacion en train-set y test-set
            cant_arg = len(arguments_by_mode[key])
            random_sample_idx = random.sample(range(cant_arg), int(cant_arg * 0.8))

            train_set = []

            for i in range(cant_arg):
                if i in random_sample_idx:
                    train_set.append(arguments_by_mode[key][i])

                else:
                    # Test set esta separado por modo de argumentacion
                    test_set[key].append(arguments_by_mode[key][i])

            train_set = np.vstack(train_set)

            # Entrenamos el modelo, calculamos vector promedio para cada modo de argumentacion
            mode_mean_vectors.append(np.mean(train_set, axis=0))

        mode_mean_vectors = np.vstack(mode_mean_vectors)

        # Calculamos el accuracy
        top1_correct = 0
        total_test_size = 0
        for key in test_set.keys():
            total_test_size += len(test_set[key])

            for arg_vec in test_set[key]:
                results = cosine_similarity(mode_mean_vectors, np.array([arg_vec]))
                results = results.reshape(1, results.size)[0]

                index = np.argsort(results)
                index_most_similar = index[-1]
                # print(mode_mean_vectors[index_most_similar][:5])
                # print(modes[index_most_similar])

                if key == modes[index_most_similar]:
                    top1_correct += 1

        print("result: " + str(top1_correct / total_test_size))
        final_accuracy += top1_correct / total_test_size

    # Calculamos el accuracy promedio
    final_accuracy = final_accuracy / repetitions
    print("Mean result: " + str(final_accuracy))

    return final_accuracy

###########################################################################################
# Clasificacion a partir de redes neuronales
###########################################################################################


