from gensim.models.keyedvectors import KeyedVectors

import numpy as np

import shutil
import torch
import os
import io
import Constant

# Clase para realizar evaluacion de word embedding segun analogias de palabras
class AnalogyTestClass:

    # Dataset y resultados
    _DATASET = Constant.DATA_FOLDER / "AnalogyDataset"

    _INTERSECT_DATASET = Constant.DATA_FOLDER / "_intersection_AnalogyDataset"
    _ORIGINAL_DATASET = Constant.DATA_FOLDER / "AnalogyDataset"

    _RESULT = Constant.RESULTS_FOLDER / "Analogy"

    """
    Inicializacion de clase.
    
    :param use_intersect_dataset: setting para utilizar la interseccion de los dataset de embeddings
    """
    def __init__(self, lower=True, use_intersect_dataset=False, datasets=[], metrics=["SpaceAnalogy", "3CosMul", "3CosAdd",]):
        print("Test de Analogias")

        self._lower = lower
        self._use_intersect_dataset = use_intersect_dataset
        self._datasets = datasets

        # "3CosMul" "3CosAdd" "SpaceAnalogy"
        self._scores_to_get = metrics

        if self._use_intersect_dataset:
            self._DATASET = self._INTERSECT_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "_intersection_Analogy"

            self.createIntersectDataset()

        else:
            self._DATASET = self._ORIGINAL_DATASET
            self._RESULT = Constant.RESULTS_FOLDER / "Analogy"

    ###########################################################################################
    # METRICAS
    ###########################################################################################

    """
    Utilizando 3CosMul, encuentra en top1 y top5 la palabra d, utilizando palabras a, b, c,
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    :return: 1 o 0, dependiendo si se encontro d en top1 y top5
    """
    def getCoseneSimilarCosmul(self, embedding, p1, p2, q1, q2):
        ans = [0, 0]

        for a in p1:
            for b in p2:
                for c in q1:
                    res = self.cosMul(embedding, a, b, c)
                    res1 = res[0][0]
                    res5 = [r[0] for r in res[:5]]

                    if res1 in q2:
                        ans[0] = 1

                    if len(list(set(res5).intersection(q2))) > 0:
                        ans[1] = 1

        return ans

    def cosMul(self, embedding, a, b, c):
        # Vectores
        v_a = torch.FloatTensor(embedding[a])
        v_b = torch.FloatTensor(embedding[b])
        v_c = torch.FloatTensor(embedding[c])

        v_a = v_a / torch.norm(v_a)
        v_b = v_b / torch.norm(v_b)
        v_c = v_c / torch.norm(v_c)

        # Lista de vectores de palabras
        vectors_matrix = torch.FloatTensor(embedding.getVectors())

        # Lista de palabras
        word_list = embedding.getWordList()

        # Distancias entre vectores
        d_a = np.dot(vectors_matrix, v_a)
        d_b = np.dot(vectors_matrix, v_b)
        d_c = np.dot(vectors_matrix, v_c)

        d_a = torch.FloatTensor(d_a)
        d_b = torch.FloatTensor(d_b)
        d_c = torch.FloatTensor(d_c)

        d_cosmul = (((d_b + 1) / 2) * ((d_c + 1) / 2)) / (((d_a + 1) / 2) + 0.000001)
        d_pos = torch.argsort(d_cosmul, descending=True)

        i = 0
        cont = 0
        res = []
        while cont < 5:
            if word_list[d_pos[i]] in [a, b, c]:
                i += 1
                continue

            res.append([word_list[d_pos[i]], d_cosmul[d_pos[i]]])
            i += 1
            cont += 1

        return res

    """
    Utilizando 3CosAdd, encuentra en top1 y top5 la palabra d, utilizando palabras a, b, c,
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    :return: 1 o 0, dependiendo si se encontro d en top1 y top5
    """
    def getCoseneSimilar(self, embedding, p1, p2, q1, q2):
        ans = [0, 0]

        for a in p1:
            for b in p2:
                for c in q1:
                    res = self.cosAdd(embedding, a, b, c)
                    res1 = res[0][0]
                    res5 = [r[0] for r in res[:5]]

                    if res1 in q2:
                        ans[0] = 1

                    if len(list(set(res5).intersection(q2))) > 0:
                        ans[1] = 1

        return ans

    def cosAdd(self, embedding, a, b, c):
        # Vectores
        v_a = torch.FloatTensor(embedding[a])
        v_b = torch.FloatTensor(embedding[b])
        v_c = torch.FloatTensor(embedding[c])

        v_a = v_a / torch.norm(v_a)
        v_b = v_b / torch.norm(v_b)
        v_c = v_c / torch.norm(v_c)

        # Lista de vectores de palabras
        vectors_matrix_norm = torch.FloatTensor(embedding.getVectors())

        # Lista de palabras
        word_list = embedding.getWordList()

        rel_vec = (v_b - v_a + v_c)
        rel_vec = rel_vec / torch.norm(rel_vec)

        d_vec = np.dot(vectors_matrix_norm, rel_vec)

        d_cosadd = torch.FloatTensor(d_vec)
        d_pos = torch.argsort(d_cosadd, descending=True)

        i = 0
        cont = 0
        res = []
        while cont < 5:
            if word_list[d_pos[i]] in [a, b, c]:
                i += 1
                continue

            res.append([word_list[d_pos[i]], d_cosadd[d_pos[i]]])
            i += 1
            cont += 1

        return res


    """
    Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia coseno como funcion de puntaje
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    :return: valor de distancia conseno entre vectores (b-a) y (d-c)
    """
    def getCos(self, embedding, p1, p2, q1, q2):
        result = -1.0
        for a in p1:
            for b in p2:
                for c in q1:
                    for d in q2:
                        r = -1.0
                        a_vec = embedding[a]
                        b_vec = embedding[b]
                        c_vec = embedding[c]
                        d_vec = embedding[d]

                        r = np.dot(b_vec - a_vec, d_vec - c_vec) / (np.linalg.norm(b_vec - a_vec)*np.linalg.norm(d_vec - c_vec))

                        if r > result:
                            result = r

        return result


    """
    Retorna maximo puntaje para la relacion a:b = c:d utilizando distancia euclidiana como funcion de puntaje
    
    :param embedding: embeddings
    :param p1: dentro de la relacion, palabras que pueden ser a
    :param p2: dentro de la relacion, palabras que pueden ser b
    :param q1: dentro de la relacion, palabras que pueden ser c
    :param q2: dentro de la relacion, palabras que pueden ser d
    
    :return: valor de distancia euclidian entre vectores (b-a) y (d-c)
    """
    def getEuc(self, embedding, p1,p2, q1, q2):
        result = -1.0
        for a in p1:
            for b in p2:
                for c in q1:
                    for d in q2:
                        a_vec = embedding[a]
                        b_vec = embedding[b]
                        c_vec = embedding[c]
                        d_vec = embedding[d]

                        r = 1 - (np.linalg.norm((b_vec - a_vec) - (d_vec - c_vec)) / (np.linalg.norm(b_vec - a_vec) + np.linalg.norm(d_vec - c_vec)))

                        if r > result:
                            result = r

        return result


    ###########################################################################################
    # MANEJO DE ARCHIVOS Y DATASET
    ###########################################################################################

    """
    Funcion que borra datasets para realizar test, el cual usa palabras de la interseccion de los vocabularios.
    """
    def resetIntersectDataset(self):
        print("Reiniciando dataset")
        intersect_dataset_path = self._INTERSECT_DATASET
        if intersect_dataset_path.exists():
            shutil.rmtree(intersect_dataset_path)

        if self._use_intersect_dataset:
            self.createIntersectDataset()


    def createIntersectDataset(self):
        print("Intersectando datasets...")

        # Verificar que existe carpeta para guardar nuevo dataset
        if not self._INTERSECT_DATASET.exists():
            os.makedirs(self._INTERSECT_DATASET)

        # Verificar si hay datasets ya intersectados
        print(" > Revisando si existe interseccion previa")
        for file_name in os.listdir(self._ORIGINAL_DATASET):
            if file_name in os.listdir(self._INTERSECT_DATASET):
                print("   > ", file_name, " ya ha sido intersectado anteriormente")
            else:
                origin_file = self._ORIGINAL_DATASET / file_name
                shutil.copy(origin_file, self._INTERSECT_DATASET)
                print("   > ", file_name, " no ha sido intersectado anteriormente, copiando")


    """
    Fucion que toma el vocabulario de un embedding y lo intersecta con el vocabulario en los dataset para el test
    :param word_vector: embedding
    :return: retorna un booleano, si hay datasets para evaluar
    """
    def intersectDataset(self, word_vector):
        print("Intersectando datasets...")
        next_dataset_path = self._INTERSECT_DATASET
        deleted_files = 0

        # Revisar cada archivo dentro de la carpeta de dataset
        print(" > Revision de archivos en dataset")
        to_delete_files = []
        for file_name in os.listdir(next_dataset_path):
            print(" > Revisando " + file_name)
            file_path = next_dataset_path / file_name
            deleted_element = 0
            lines = []

            if "4tupla" in str(file_path):
                print("   Revisando archivo de tipo 4-tupla")

            # Revisar el dataset intersectado que llevamos hasta el momento
            with io.open(file_path, 'r') as f:
                for line in f:
                    tupla = line.lower().split() if self._lower else line.split()

                    p1 = tupla[0].split('/')
                    p2 = tupla[1].split('/')

                    # Caso de archivo con 4-tuplas
                    p3 = tupla[2].split('/') if "4tupla" in str(file_path) else []
                    p4 = tupla[3].split('/') if "4tupla" in str(file_path) else []

                    q1 = []
                    q2 = []
                    q3 = []
                    q4 = []

                    for p in p1:
                        if p in word_vector:
                            q1.append(p)

                    for p in p2:
                        if p in word_vector:
                            q2.append(p)

                    for p in p3:
                        if p in word_vector:
                            q3.append(p)

                    for p in p4:
                        if p in word_vector:
                            q4.append(p)

                    # Verificar que todas la palabras en las analogias estan presentes
                    if len(q1) == 0 or len(q2) == 0:
                        deleted_element += 1
                        continue

                    if ("4tupla" in str(file_path)) and (len(q3) == 0 or len(q4) == 0):
                        deleted_element += 1
                        continue

                    # Transforma tupla completa a string
                    line = '/'.join(q1) + "\t" + '/'.join(q2) + "\n"
                    if "4tupla" in str(file_path):
                        line = '/'.join(q1) + "\t" + '/'.join(q2) + "\t" + '/'.join(q3) + "\t" + '/'.join(q4) + "\n"

                    lines.append(line)

            # Caso en que el archivo queda vacio
            if len(lines) == 0:
                deleted_files += 1
                to_delete_files.append(file_path)
                print(" > Archivo esta vacio, se procede a eliminar")
                continue

            # Escribir la nueva interseccion
            with io.open(file_path, 'w') as f:
                for line in lines:
                    f.write(line)

            print(" > Lineas eliminadas: " + str(deleted_element) + " de " + str(deleted_element + len(lines)))

        print(" > Archivos a eliminar: " + str(deleted_files))
        for file in to_delete_files:
            os.remove(file)

        return True if len(os.listdir(next_dataset_path)) > 0 else False



    """
    Obtencion de nombre de los distintos archivos de test de analogias
    :return: lista con path completo de los distintos archivos con pares de palabras para test de analogias
    """
    def getTestFiles(self):
        dataset_folder = self._DATASET

        if not dataset_folder.exists():
            raise Exception("No se logro encontrar carpeta con test")

        test_files = []

        for file_name in os.listdir(dataset_folder):
            use = (len(self._datasets) == 0)
            for set_name in self._datasets:
                if set_name in file_name:
                    use = True

            if use:
                test_files.append(dataset_folder / file_name)

        return test_files


    """
    Obtencion de path completo hacia los dintintos archivos de test de analogias que no han sido evaluados
    :param embedding_name: nombre del embedding que se va a evaluar
    :return: lista de archivos a usar en evaluacion
    """
    def getUntestedFiles(self, embedding_name):
        print(">>> Buscando archivos sin testear")
        test_files = self.getTestFiles()
        result_path = self._RESULT / embedding_name

        # Revisar que existe la carpeta de resultados parciales
        if not result_path.exists():
            return test_files

        # Eliminar archivos que ya han sido utilizados en evaluacion
        print(">>> Test files encontrados")
        test_files_list = []
        for file in test_files:
            print("   ", str(file))
            # Path hacia el resultado del test asociado al archivo file
            temp_result_file_path = result_path / file.name

            if not temp_result_file_path.exists():
                test_files_list.append(file)
            else:
                print("    > Ya existe un resultados con este test")

        return test_files_list


    """
    Obtencion de los pares de palabras (a:b) presentes en un archivo de test
    :param test_file_path: path hacia archivo con pares de palabras
    :param lower: determina si las palabras solo se consideran en minusculas
    :return: lista de pares de palabras 
    """
    def getAnalogyPairs(self, test_file_path):
        if not test_file_path.exists():
            raise Exception("No existe archivo pedido")

        word_pair = []
        with io.open(test_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.lower() if self._lower else line

                pair = line.split()
                word_pair.append(pair)

        return word_pair


    '''
    Eliminacion de palabras oov dentro de una tupla
    :param embedding: lista con vectores de palabras
    :param p1: lista de palabras que representa el elemento a en analogia
    :param p2: lista de palabras que representa el elemento b en analogia
    :param q1: lista de palabras que representa el elemento c en analogia
    :param q2: lista de palabras que representa el elemento d en analogia
    :return: tupla con lista de palabras, cada lista representa un elemento en la analogia
    '''
    def delete_oov(self, embedding, p1, p2, q1, q2):
        number_of_oov_element = 0
        new_tuple = []

        for tuple in [p1, p2, q1, q2]:
            r = []

            for word in tuple:
                if word not in embedding:
                    continue

                r.append(word)

            if len(r) == 0:
                number_of_oov_element += 1

            new_tuple.append(r)

        return number_of_oov_element, new_tuple


    ###########################################################################################
    # GUARDAR RESULTADOS
    ###########################################################################################

    def cleanResults(self):
        print("Limpiando resultados temporales... ", end='')
        temp_result_path = self._RESULT
        if temp_result_path.exists():
            shutil.rmtree(temp_result_path)

        print("listo.")

    """
    Guarda resultados de analogias de forma temporal
    
    :param embedding_name: nombre del embedding que se testeo
    :param test_file_name: nombre del archivo que contiene los pares de palabras usados en el test
    :param results_list: resultados del test sobre distintas metricas, pares (nombre test, resultado)
    """
    def saveResults(self, embedding_name, test_file_name, results_list):
        temp_result_embedding = self._RESULT / embedding_name

        if not temp_result_embedding.exists():
            os.makedirs(temp_result_embedding)

        temp_result_file = temp_result_embedding / test_file_name

        with io.open(temp_result_file, 'w') as f:
            for res in results_list:
                print(res)
                for word in res:
                    f.write(str(word) + " ")

                f.write("\n")


    """
    Junta todos los resultados de un embedding y luego los guarda en un mismo archivo de resultados.
    
    :param embedding_name: nombre del embedding testeado
    
    :return: resultado de cada uno de los archivos de test, ademas de informacion relacionada a la cantidad de analogias no evaluadas
    """
    def getAllResults(self, embedding_name):
        temp_analogy_results_folder = self._RESULT
        temp_result_embedding = temp_analogy_results_folder / embedding_name


        # Revisar que existe la carpeta de resultado temporales
        if not temp_result_embedding.exists():
            raise Exception("Falta carpeta con resultados temporales, no se pueden obtener resultados")


        # Extraccion de resultados temporales
        test_result_list = os.listdir(temp_result_embedding)
        results = []
        for test_file_name in test_result_list:
            test_result_file = temp_result_embedding / test_file_name

            aux_result = []
            with io.open(test_result_file, 'r') as f:
                for line in f:
                    aux_result.append(line.strip().split())

            results.append([test_file_name, aux_result])

        print(">>> Resultados")
        for r in results:
            print("    ", end='')
            print(r)

        print('')

        return results


    ###########################################################################################
    # EVALUACION POR ANALOGIAS
    ###########################################################################################

    def CosMulResults(self, file, pair_list, word_vector):
        if "4tupla" not in str(file):
            tuple_list = []

            for i in range(len(pair_list)):
                for j in range(len(pair_list)):
                    if(i == j):
                        continue

                    p1 = pair_list[i][0]
                    p2 = pair_list[i][1]
                    q1 = pair_list[j][0]
                    q2 = pair_list[j][1]

                    tuple_list.append([p1, p2, q1, q2])

            pair_list = tuple_list

        print(" > Cantidad de analogias a evaluar: ", str(len(pair_list)))

        oov_tuples = 0
        analogy_count = 0
        top1_count = 0
        top5_count = 0

        for tupla in pair_list:
            p1 = tupla[0].strip().split('/')
            p2 = tupla[1].strip().split('/')
            q1 = tupla[2].strip().split('/')
            q2 = tupla[3].strip().split('/')

            number_of_oov_elements, new_tuple = self.delete_oov(word_vector, p1, p2, q1, q2)
            p1 = new_tuple[0]
            p2 = new_tuple[1]
            q1 = new_tuple[2]
            q2 = new_tuple[3]

            if number_of_oov_elements > 0:
                oov_tuples += 1
                continue

            analogy_count += 1

            top1, top5 = self.getCoseneSimilarCosmul(word_vector, p1, p2, q1, q2)
            top1_count += top1
            top5_count += top5

        top1_res = top1_count / analogy_count if analogy_count > 0 else 0
        top5_res = top5_count / analogy_count if analogy_count > 0 else 0

        return [["top1_3CosMul", top1_res], ["top5_3CosMul", top5_res]]


    def CosAddResults(self, file, pair_list, word_vector):
        if "4tupla" not in str(file):
            tuple_list = []

            for i in range(len(pair_list)):
                for j in range(len(pair_list)):
                    if(i == j):
                        continue

                    p1 = pair_list[i][0]
                    p2 = pair_list[i][1]
                    q1 = pair_list[j][0]
                    q2 = pair_list[j][1]

                    tuple_list.append([p1, p2, q1, q2])

            pair_list = tuple_list

        print(" > Cantidad de analogias a evaluar: ", str(len(pair_list)))

        oov_tuples = 0
        analogy_count = 0
        top1_count = 0
        top5_count = 0

        for tupla in pair_list:
            p1 = tupla[0].strip().split('/')
            p2 = tupla[1].strip().split('/')
            q1 = tupla[2].strip().split('/')
            q2 = tupla[3].strip().split('/')

            number_of_oov_elements, new_tuple = self.delete_oov(word_vector, p1, p2, q1, q2)
            p1 = new_tuple[0]
            p2 = new_tuple[1]
            q1 = new_tuple[2]
            q2 = new_tuple[3]

            if number_of_oov_elements > 0:
                oov_tuples += 1
                continue

            analogy_count += 1

            top1, top5 = self.getCoseneSimilar(word_vector, p1, p2, q1, q2)
            top1_count += top1
            top5_count += top5

        top1_res = top1_count / analogy_count if analogy_count > 0 else 0
        top5_res = top5_count / analogy_count if analogy_count > 0 else 0

        return [["top1_3CosAdd", top1_res], ["top5_3CosAdd", top5_res]]

    def AnalogySpaceResults(self, file, pair_list, word_vector):
        if "4tupla" not in str(file):
            tuple_list = []

            for i in range(len(pair_list)):
                for j in range(len(pair_list)):
                    if j <= i:
                        continue

                    p1 = pair_list[i][0]
                    p2 = pair_list[i][1]
                    q1 = pair_list[j][0]
                    q2 = pair_list[j][1]

                    tuple_list.append([p1, p2, q1, q2])

            pair_list = tuple_list

        print(" > Cantidad de analogias a evaluar: ", str(len(pair_list)))

        oov_tuples = 0
        analogy_count = 0
        cos_count = 0
        euc_count = 0

        for tupla in pair_list:
            p1 = tupla[0].strip().split('/')
            p2 = tupla[1].strip().split('/')
            q1 = tupla[2].strip().split('/')
            q2 = tupla[3].strip().split('/')

            number_of_oov_elements, new_tuple = self.delete_oov(word_vector, p1, p2, q1, q2)
            p1 = new_tuple[0]
            p2 = new_tuple[1]
            q1 = new_tuple[2]
            q2 = new_tuple[3]

            if number_of_oov_elements > 0:
                oov_tuples += 1
                continue

            analogy_count += 1

            res_cos = self.getCos(word_vector, p1, p2, q1, q2)
            cos_count += res_cos

            res_euc = self.getEuc(word_vector, p1, p2, q1, q2)
            euc_count += res_euc

        cos_res = cos_count / analogy_count if analogy_count > 0 else 0
        euc_res = euc_count / analogy_count if analogy_count > 0 else 0

        return [["Cos", cos_res], ["Euc", euc_res]]

    """
    Evalua un word embedding especifico y guarda el resultado en carpeta de resultados

    :param word_vector_name: nombre de word embedding
    :param word_vector: word embedding a evaluar
    """
    def evaluateWordVector(self, word_vector_name, word_vector):
        # Obtencion de archivos que faltan por testear
        test_file_list = self.getUntestedFiles(word_vector_name)
        print("Untested files:")
        for file in test_file_list:
            print(">", file)

        # Revisamos todos los archivos para realizar test
        for file in test_file_list:
            file_results = self.evaluateFile(file, word_vector)

            # Guardamos los resultados
            self.saveResults(word_vector_name, file.name, file_results)


    """
    Obtencion de distintos resultados en evaluacion por analogias
    
    :param file: nombre de carpeta con analogias a utilizar
    :param word_vector: word embedding a evaluar
    """
    def evaluateFile(self, file, word_vector):
        print(">>> Testing: ", file.name)
        pair_list = self.getAnalogyPairs(file)

        res = []
        if "3CosMul" in self._scores_to_get:
            res_cosmul = self.CosMulResults(file, pair_list, word_vector)
            res = res + res_cosmul

        if "3CosAdd" in self._scores_to_get:
            res_cosadd = self.CosAddResults(file, pair_list, word_vector)
            res = res + res_cosadd

        if "SpaceAnalogy" in self._scores_to_get:
            res_aspace = self.AnalogySpaceResults(file, pair_list, word_vector)
            res = res + res_aspace

        return res
