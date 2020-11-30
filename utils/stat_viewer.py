from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

inverse = True
implication = False
symmetric = False
one_to_many = False
reflexive = False

# For reflexive creation
def write_to_txt_file(path, data):
    """
     :param

     path: path where to be saved
     data: triples to be written in txt


     """
    f = open(path, "w")
    for i in range(data.shape[0]):
        line = ''
        for j in range(data.shape[1]):
            if (j == 0):
                line = str(data[i][j])
            else:
                line = line + '\t' + str(data[i][j])
        f.write(line)
        f.write("\n")
        # print(line)
    f.close()


dirname = os.path.dirname(__file__)
dataset = "fb15k-237"
relPathToData = "../data/"

data_dir = os.path.join(dirname, relPathToData, dataset)

pathToTrainTriples = os.path.join(data_dir, "train.txt")
all_training_triple = pd.read_table(pathToTrainTriples, header=None)

pathToTestTriples = os.path.join(data_dir, "test.txt")
all_test_triple = pd.read_table(pathToTestTriples, header=None)

original = np.array([])
symmtetric_var = np.array([])
original_list = []
reflexive_list = []
all_training_triple_other = all_training_triple.copy()


###########################################for reflexive################################################

def analyse_reflexive():
    reflexive_triples = all_training_triple.loc[all_training_triple[0] == all_training_triple[2]]
    reflexive_triples = pd.DataFrame(np.array(reflexive_triples))
    unique_relations_reflexive_triples = np.array(list(set(list(reflexive_triples[1]))))
    reflexive_triple_per_relation = []
    for unique_relations_reflexive_triple in unique_relations_reflexive_triples:
        place_holder_relation_triple = reflexive_triples.loc[reflexive_triples[1] == unique_relations_reflexive_triple]
        place_holder_relation_triple = np.array(place_holder_relation_triple)
        stat_count_per_relation = np.array([unique_relations_reflexive_triple, int(len(place_holder_relation_triple))])
        reflexive_triple_per_relation.append(stat_count_per_relation)
    reflexive_triple_per_relation = pd.DataFrame(np.array(reflexive_triple_per_relation))
    reflexive_triple_per_relation[1] = reflexive_triple_per_relation[1].astype(str).astype(int)
    reflexive_triples_per_relation_sorted = reflexive_triple_per_relation.sort_values(by=1, ascending=False)
    # count the number of triples in  the test set
    reflexive_relation_array = np.array(reflexive_triples_per_relation_sorted[0])
    count_list_reflexive = np.zeros((0, 3))
    for reflexive_relation in reflexive_relation_array:
        place_holder_relation_test_triple = all_test_triple.loc[all_test_triple[1] == reflexive_relation]
        place_holder_relation_test_triple_array = np.array(place_holder_relation_test_triple)
        count_list_reflexive = np.vstack([count_list_reflexive, place_holder_relation_test_triple_array])
    print('total number of ERE exist in the training set: ')
    print(len(reflexive_triples))
    print('ERE per relation exist in the training set: ')
    print(reflexive_triples_per_relation_sorted)
    print('for each R in the training set for which ERE exist, how many triple exist in test set: ')
    print(len(count_list_reflexive))
    # Save such test triples
    write_dir = os.path.join(data_dir, 'reflexive.txt')
    write_to_txt_file(write_dir, count_list_reflexive)



def analyse_symmetric():
    ###########################################for Symmetric################################################
    training_triples = list(zip([h for h in all_training_triple[0]],
                                [r for r in all_training_triple[1]],
                                [t for t in all_training_triple[2]]))
    premise_list = []
    conclusion_list = []
    observed_triples = {}
    for triple in training_triples:
        observed_triples[triple[0], triple[1], triple[2]] = 1
    # for symmetric
    for triple in training_triples:
        try:
            exists = observed_triples[triple[2], triple[1], triple[0]]
            print('original_triple ', triple)
            premise = np.array([triple[0], triple[1], triple[2]])
            conclusion = np.array([triple[2], triple[1], triple[0]])
            print('premise ', premise)
            print('conclusion ', conclusion)
            print('######################################')
            premise_list.append(np.array(premise))
            conclusion_list.append(np.array(conclusion))
        except:
            continue
    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))

    symmetric_triple_per_relation = []
    for premise in premise_df_unique_relation:
        symmetric_array_place_holder = premise_df.loc[premise_df[1] == premise]
        symmetric_array_place_holder = np.array(symmetric_array_place_holder)
        stat_count_per_relation_symmetric = np.array(
            [premise, int(len(symmetric_array_place_holder))])
        symmetric_triple_per_relation.append(stat_count_per_relation_symmetric)
    symmetric_triple_per_relation = pd.DataFrame(np.array(symmetric_triple_per_relation))
    symmetric_triple_per_relation[1] = symmetric_triple_per_relation[1].astype(str).astype(int)
    symmetric_triple_per_relation_sorted = symmetric_triple_per_relation.sort_values(by=1, ascending=False)
    # in the test set:
    symmetric_relation_array = np.array(symmetric_triple_per_relation_sorted[0])
    count_list_symmetric = np.zeros((0, 3))
    for symmetric_relation in symmetric_relation_array:
        place_holder_relation_test_triple = all_test_triple.loc[all_test_triple[1] == symmetric_relation]
        place_holder_relation_test_triple_array = np.array(place_holder_relation_test_triple)
        count_list_symmetric = np.vstack([count_list_symmetric, place_holder_relation_test_triple_array])
    print('total number of symmetric exist in the training set: ')
    print(len(premise_df))
    print('symmetric per relation exist in the training set: ')
    print(symmetric_triple_per_relation_sorted)
    pathToSymmetricStats = os.path.join(data_dir, "symmetric_stats_train.xlsx")
    symmetric_triple_per_relation_sorted.to_excel(pathToSymmetricStats, index=False)
    plt.figure()
    symmetric_triple_per_relation_sorted.plot(kind='bar')
    plotSavePath = os.path.join(data_dir, "symmetric-plot.png")
    plt.gcf().savefig(plotSavePath)
    plt.show()
    print('for each premise R in the training set for which symmetric exist, how many triple exist in test set: ')
    print(len(count_list_symmetric))
    write_dir = os.path.join(data_dir, 'symmetric.txt')
    write_to_txt_file(write_dir, count_list_symmetric)



def analyse_one_to_many():
    #######################################For one-to-many###################################################################
    all_relations_train = list(set(list(all_training_triple[1])))
    summarizeInfo = pd.DataFrame(columns=["relation", "min", "max", "avg", "count"])
    premise_list = []
    conclusion_list = []
    training_triples = list(zip([h for h in all_training_triple[0]],
                                [r for r in all_training_triple[1]],
                                [t for t in all_training_triple[2]]))
    observed_triples = {}
    for triple in training_triples:
        observed_triples[triple[0], triple[1], triple[2]] = 1
    count = 0
    for relation in all_relations_train:
        the_N_for_each_h_and_r: List[int] = []
        for idx, triple in enumerate(training_triples):
            count = 0
            if triple != ('', '', ''):
                if triple[1] == relation:
                    # relation = triple[1]
                    for idx2, triple2 in enumerate(training_triples):
                        if triple[0] == triple2[0] and triple2[1] == relation and triple[2] != triple2[2]:
                            training_triples[idx2] = ('', '', '')
                            count += 1
                    # if relation in relation_count.keys() and count != 0:
                    #     relation_count[relation] += count
                    # elif count != 0:
                    #     relation_count[relation] = count
                    if count != 0:
                        the_N_for_each_h_and_r.append(count)
        if len(the_N_for_each_h_and_r) > 0:
            minCount = min(the_N_for_each_h_and_r)
            maxCount = max(the_N_for_each_h_and_r)
            avg = sum(the_N_for_each_h_and_r) / len(the_N_for_each_h_and_r)
            avg = round(avg, 2)
            total_count = sum(the_N_for_each_h_and_r)
        else:
            minCount = maxCount = avg = total_count = 0
        frame = pd.DataFrame([[relation, minCount, maxCount, avg, total_count]],
                             columns=["relation", "min", "max", "avg", "count"])
        summarizeInfo = summarizeInfo.append(frame)
        total_count = 0
    # one_to_many_triple_per_relation = pd.DataFrame(list(relation_count.items()),columns=['relation', 'number of one-to-many relatioins'])
    one_to_many_triple_per_relation = summarizeInfo
    pathToInverseStats = os.path.join(data_dir, "one_to_many_stats_train.xlsx")
    one_to_many_triple_per_relation.to_excel(pathToInverseStats, index=False, columns=False)


def analyse_implication():
    #######################################For implication###################################################################

    all_relations_train = list(set(list(all_training_triple[1])))

    premise_list = []
    conclusion_list = []
    training_triples = list(zip([h for h in all_training_triple[0]],
                                [r for r in all_training_triple[1]],
                                [t for t in all_training_triple[2]]))
    observed_triples = {}
    for triple in training_triples:
        observed_triples[triple[0], triple[1], triple[2]] = 1
    count = 0
    # for implication
    for triple in training_triples:
        print(count)
        for r2 in all_relations_train:
            if (r2 != triple[1]):
                try:
                    exists = observed_triples[triple[0], r2, triple[2]]
                    premise = np.array([triple[0], triple[1], triple[2]])
                    conclusion = np.array([triple[0], r2, triple[2]])
                    print('premise ', premise)
                    print('conclusion ', conclusion)
                    print('######################################')
                    premise_list.append(np.array(premise))
                    conclusion_list.append(np.array(conclusion))
                except:
                    continue
        count += 1

    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))
    implication_triple_per_relation = []
    for premise in premise_df_unique_relation:
        implication_array_place_holder = premise_df.loc[premise_df[1] == premise]
        implication_array_place_holder = np.array(implication_array_place_holder)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation_implication = np.array(
            [premise, int(len(implication_array_place_holder))])
        implication_triple_per_relation.append(stat_count_per_relation_implication)
    implication_triple_per_relation = pd.DataFrame(np.array(implication_triple_per_relation))
    implication_triple_per_relation[1] = implication_triple_per_relation[1].astype(str).astype(int)
    implication_triple_per_relation_sorted = implication_triple_per_relation.sort_values(by=1, ascending=False)

    # In the test set
    implication_relation_array = np.array(implication_triple_per_relation_sorted[0])
    count_list_implication = np.zeros((0, 3))
    for implication_relation in implication_relation_array:
        place_holder_relation_test_triple = all_test_triple.loc[all_test_triple[1] == implication_relation]
        place_holder_relation_test_triple_array = np.array(place_holder_relation_test_triple)
        count_list_implication = np.vstack([count_list_implication, place_holder_relation_test_triple_array])

    print('total number of implication exist in the training set: ')
    print(len(premise_df))
    print('implication per relation exist in the training set: ')
    print(implication_triple_per_relation_sorted)
    pathToImplicationStats = os.path.join(data_dir, "implication_stats_train.xlsx")
    implication_triple_per_relation_sorted.to_excel(pathToImplicationStats, index=False)
    plt.figure()
    implication_triple_per_relation_sorted.plot(kind='bar')
    plotSavePath = os.path.join(data_dir, "implication-plot.png")
    plt.gcf().savefig(plotSavePath)
    plt.show()
    print('for each premise R in the training set for which implication exist, how many triple exist in test set: ')
    print(len(count_list_implication))

    write_dir = os.path.join(data_dir, 'implication.txt')
    write_to_txt_file(write_dir, count_list_implication)



def analyse_inverse():
    #################################Inverse##########################################################################

    all_relations_train = list(set(list(all_training_triple[1])))
    premise_list = []
    conclusion_list = []
    training_triples = list(zip([h for h in all_training_triple[0]],
                                [r for r in all_training_triple[1]],
                                [t for t in all_training_triple[2]]))
    observed_triples = {}
    for triple in training_triples:
        observed_triples[triple[0], triple[1], triple[2]] = 1
    # for inverse
    count = 0
    # for implication
    for triple in training_triples:
        print(count)
        for r2 in all_relations_train:
            if (r2 != triple[1]):
                try:
                    exists = observed_triples[triple[2], r2, triple[0]]
                    premise = np.array([triple[0], triple[1], triple[2]])
                    conclusion = np.array([triple[2], r2, triple[0]])
                    print('premise ', premise)
                    print('conclusion ', conclusion)
                    print('######################################')
                    premise_list.append(np.array(premise))
                    conclusion_list.append(np.array(conclusion))
                except:
                    continue
        count += 1

    premise_df = pd.DataFrame(np.array(premise_list)).applymap(str)
    conclusion_df = pd.DataFrame(np.array(conclusion_list)).applymap(str)
    premise_df_unique_relation = np.array(list(set(list(premise_df[1]))))
    conclusion_df_unique_relation = np.array(list(set(list(conclusion_df[1]))))
    inverse_triple_per_relation = []

    for premise in premise_df_unique_relation:
        inverse_array_place_holder = premise_df.loc[premise_df[1] == premise]
        inverse_array_place_holder = np.array(inverse_array_place_holder)
        # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
        stat_count_per_relation_inverse = np.array(
            [premise, int(len(inverse_array_place_holder))])
        inverse_triple_per_relation.append(stat_count_per_relation_inverse)
    inverse_triple_per_relation = pd.DataFrame(np.array(inverse_triple_per_relation))
    inverse_triple_per_relation[1] = inverse_triple_per_relation[1].astype(str).astype(int)
    inverse_triple_per_relation_sorted = inverse_triple_per_relation.sort_values(by=1, ascending=False)
    # In the test set
    inverse_relation_array = np.array(inverse_triple_per_relation_sorted[0])
    count_list_inverse = np.zeros((0, 3))

    for inverse_relation in inverse_relation_array:
        place_holder_relation_test_triple = all_test_triple.loc[all_test_triple[1] == inverse_relation]
        place_holder_relation_test_triple_array = np.array(place_holder_relation_test_triple)
        count_list_inverse = np.vstack([count_list_inverse, place_holder_relation_test_triple_array])

    print('total number of inverse exist in the training set: ')
    print(len(premise_df))
    print('inverse per relation exist in the training set: ')
    print(inverse_triple_per_relation_sorted)
    pathToInverseStats = os.path.join(data_dir, "inverse_stats_train.xlsx")
    inverse_triple_per_relation_sorted.to_excel(pathToInverseStats, index=False)
    plt.figure()
    inverse_triple_per_relation_sorted.plot(kind='bar')
    plotSavePath = os.path.join(data_dir, "inverse-plot.png")
    plt.gcf().savefig(plotSavePath)
    plt.show()
    print('for each premise R in the training set for which inverse exist, how many triple exist in test set: ')
    count_list_inverse_df = pd.DataFrame(np.array(count_list_inverse))
    print(count_list_inverse_df.sort_values(by=1, ascending=False))

    write_dir = os.path.join(data_dir, 'inverse.txt')
    write_to_txt_file(write_dir, count_list_inverse)



if reflexive:
    analyse_reflexive()
if inverse:
    analyse_inverse()
if implication:
    analyse_implication()
if symmetric:
    analyse_symmetric()
if one_to_many:
    analyse_one_to_many()
