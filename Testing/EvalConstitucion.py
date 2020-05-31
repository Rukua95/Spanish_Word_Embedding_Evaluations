from sklearn.metrics.pairwise import cosine_similarity

import ConstitucionUtil
import numpy as np

import random


def MeanVectorEvaluation(word_vector, word_vector_name):
    # Obtencion de datos ordenados, ademas de sus respectivos vectores promedios.
    gob_concept_vectors, gob_args_vectors, open_args_vectors, mode_vectors = ConstitucionUtil.getSortedDataset(word_vector)


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
    """
    Por concepto en cada topico, eligo el 80% como train y 20% como test, y unas cuantas repeticiones.
    """

    total = 0
    total_evaluado = 0
    top1_correct = 0
    args_by_concept_by_topic = {}

    # Organizar argumentos por topico y por concepto
    for topic in gob_args_vectors.keys():
        print("Topico: " + topic)

        if topic not in args_by_concept_by_topic.keys():
            args_by_concept_by_topic[topic] = {}

        total += len(gob_args_vectors[topic])

        for tupla in gob_args_vectors[topic]:
            concept = gob_args_vectors[topic]["concept"]["content"]
            arg_vector = gob_args_vectors[topic]["arg"]["vector"]

            # Omitimos vectores vacios
            if arg_vector.size == 0:
                continue

            total_evaluado += 1

            if concept not in args_by_concept_by_topic[topic].keys():
                args_by_concept_by_topic[topic][concept] = []

            args_by_concept_by_topic[topic][concept].append(arg_vector)

        for concept in args_by_concept_by_topic[topic].keys():
            print(" > " + concept + ", cantidad: " + str(len(args_by_concept_by_topic[topic][concept])))


    # Pruebas
    repetitions = 1
    final_result = {}
    for h in range(repetitions):
        print("Experimento " + str(h+1))
        for topic in args_by_concept_by_topic.keys():
            print(" > " + topic)

            test_set = {}
            model = []
            concept_list = []

            # Separamos train y test set, y entrenamos el modelo
            for concept in args_by_concept_by_topic[topic].keys():
                total_args = len(args_by_concept_by_topic[topic][concept])
                print("    > " + concept + ", cantidad: " + str(total_args))

                idx = random.sample(range(total_args), int(total_args*0.8))

                train_set = []

                for i in range(total_args):
                    if i in idx:
                        train_set.append(args_by_concept_by_topic[topic][concept][i])
                    else:
                        if concept not in test_set.keys():
                            test_set[concept] = []

                        test_set[concept].append(args_by_concept_by_topic[topic][concept][i])

                train_set = np.vstack(train_set)
                model.append(np.mean(train_set, axis=0))
                concept_list.append(concept)

            # Realizamos test
            total_test_size = 0
            for concept in test_set.keys():
                total_test_size += len(test_set[concept])

                for arg_vec in test_set[concept]:
                    results = cosine_similarity(model, np.array([arg_vec]))
                    results = results.reshape(1, results.size)[0]

                    index = np.argsort(results)
                    index_most_similar = index[-1]
                    # print(mode_mean_vectors[index_most_similar][:5])
                    # print(modes[index_most_similar])

                    if concept == concept_list[index_most_similar]:
                        top1_correct += 1

            print(" > topic " + topic + " results")
            print(top1_correct/total_test_size)

            if topic not in final_result.keys():
                final_result[topic] = 0

            final_result[topic] += top1_correct/total_test_size

    for key in final_result.keys():
        final_result[key] = final_result[key]/repetitions

    return final_result


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
            print(" > " + concept + ": ", end='')
            print(gob_concept_vectors[topic][concept].shape)

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

    for topic in gob_concept_vectors_list_by_topics.keys():
        print(topic)
        print(gob_concept_vectors_list_by_topics[topic])

    ######################################################################################

    total = 0
    total_evaluado = 0
    top5_correct = 0
    top1_correct = 0

    acuraccy_results = []


    # Cantidad de argumentos por topico
    total_open_args = 0
    for topic in open_args_vectors.keys():
        print(topic + " " + str(len(open_args_vectors[topic])))
        total_open_args += len(open_args_vectors[topic])

    print("total: " + str(total_open_args) + "\n")


    # Obtencion accuracy (top1 y top5) de similaridad.
    for topic in open_args_vectors.keys():
        print(topic + ") cantidad " + str(len(open_args_vectors[topic])))

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

            # print("###############")
            # print("top1: " + gob_concept_list_by_topics[topic][index_most_similar])
            # for id in index_most_similar_top5:
            #    print("top5: " + gob_concept_list_by_topics[topic][id])

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

        print(str(top1_correct) + " " + str(top5_correct))
        print(str(top1_acuraccy) + " " + str(top5_acuraccy))

        acuraccy_results.append([top1_acuraccy, top5_acuraccy, total_evaluado, total])

        total = 0
        total_evaluado = 0
        top5_correct = 0
        top1_correct = 0


    total_final = 0
    for r in acuraccy_results:
        print(r)
        total_final += r[-1]
        print("% omitidos: " + str(1 - r[-2] / r[-1]))
    print("total final: " + str(total_final))

    #saveResults(acuraccy_results)

    return acuraccy_results



def meanVectorEvalTaskC(gob_args_vectors, open_args_vectors):
    total = 0
    total_evaluado = 0

    # Separamos los argumentos segun modo de argumentacion
    arguments_by_mode = {}
    for key in gob_args_vectors.keys():
        total += len(gob_args_vectors[key] + open_args_vectors[key])

        for tupla in gob_args_vectors[key] + open_args_vectors[key]:
            arg_vec = tupla["arg"]["vector"]
            mode = tupla["mode"]

            if arg_vec.size == 0:
                continue

            if mode == "undefined" or mode == "blank":
                continue

            total_evaluado += 1

            if mode not in arguments_by_mode.keys():
                arguments_by_mode[mode] = []

            arguments_by_mode[mode].append(arg_vec)

    print("total de argumentos: " + str(total))
    print("total a evaluar: " + str(total_evaluado))

    for mode in arguments_by_mode.keys():
        print(mode + ", cantidad de argumentos: " + str(len(arguments_by_mode[mode])))


    # Realizamos los experimentos
    final_accuracy = 0
    repetitions = 100
    for h in range(repetitions):
        print("Experiment " + str(h+1))
        test_set = {}
        mode_mean_vectors = []
        modes = []
        top1_correct = 0

        for key in arguments_by_mode.keys():
            #print("mode: " + key)
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
                    test_set[key].append(arguments_by_mode[key][i])

            train_set = np.vstack(train_set)

            # Entrenamos el modelo, calculamos vector promedio para cada modo de argumentacion
            mode_mean_vectors.append(np.mean(train_set, axis=0))

            #print(np.mean(train_set, axis=0)[:5])

        mode_mean_vectors = np.vstack(mode_mean_vectors)

        #print("Tamaño de test set:")
        #for key in test_set.keys():
        #    print(" > " + key + ": " + str(len(test_set[key])))

        # Calculamos el accuracy
        total_test_size = 0
        for key in test_set.keys():
            #print("Evaluating arguments of mode " + key)
            total_test_size += len(test_set[key])
            for arg_vec in test_set[key]:
                results = cosine_similarity(mode_mean_vectors, np.array([arg_vec]))
                results = results.reshape(1, results.size)[0]

                index = np.argsort(results)
                index_most_similar = index[-1]
                #print(mode_mean_vectors[index_most_similar][:5])
                #print(modes[index_most_similar])

                if key == modes[index_most_similar]:
                    top1_correct += 1

        print("top1_correct: " + str(top1_correct))
        print("result: " + str(top1_correct/total_test_size))
        final_accuracy += top1_correct/total_test_size

    # Calculamos el accuracy promedio
    final_accuracy = final_accuracy/repetitions

    return final_accuracy