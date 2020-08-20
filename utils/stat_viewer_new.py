import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter
from itertools import combinations
import os


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
dataset = "wn18rr"
relPathToData = "../data/"

data_dir = os.path.join(dirname, relPathToData, dataset)
# data_dir = '/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18rr'
# all_training_triple= pd.read_csv('/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18rr/train.txt', dtype=str, header=None, sep='\t')
# all_test_triple = pd.read_csv('/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18rr/test.txt', dtype=str, header=None, sep='\t')


pathToTrainTriples = os.path.join(data_dir, "train.txt")
all_training_triple = pd.read_table(pathToTrainTriples, header=None)
# all_training_triple = pd.read_table('/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18rr/train.txt', header=None)

pathToTestTriples = os.path.join(data_dir, "test.txt")
all_test_triple = pd.read_table(pathToTestTriples, header=None)
# all_test_triple = pd.read_table('/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18rr/test.txt', header=None)

original = np.array([])
symmtetric_var = np.array([])
original_list = []
reflexive_list = []
all_training_triple_other = all_training_triple.copy()
# count = 0

# for reflexive
reflexive_triples = all_training_triple.loc[all_training_triple[0] == all_training_triple[2]]
reflexive_triples = pd.DataFrame(np.array(reflexive_triples))
unique_relations_reflexive_triples = np.array(list(set(list(reflexive_triples[1]))))

# reflexive_triples = np.array(reflexive_triples)


reflexive_triple_per_relation = []
for unique_relations_reflexive_triple in unique_relations_reflexive_triples:
    # print(unique_relations_reflexive_triple)
    place_holder_relation_triple = reflexive_triples.loc[reflexive_triples[1] == unique_relations_reflexive_triple]
    place_holder_relation_triple = np.array(place_holder_relation_triple)
    # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
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
# data_dir = '/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18'
write_dir = os.path.join(data_dir, 'reflexive.txt')
write_to_txt_file(write_dir, count_list_reflexive)

###########################################for Symmetric################################################
symmetric_list = []

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
        # symmetric = np.array([triple[2], triple[1], triple[0]])
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

#####################################################################################################
symmetric_triple_per_relation = []
for premise in premise_df_unique_relation:
    symmetric_array_place_holder = premise_df.loc[premise_df[1] == premise]
    symmetric_array_place_holder = np.array(symmetric_array_place_holder)
    # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
    stat_count_per_relation_symmetric = np.array(
        [premise, int(len(symmetric_array_place_holder))])
    symmetric_triple_per_relation.append(stat_count_per_relation_symmetric)

symmetric_triple_per_relation = pd.DataFrame(np.array(symmetric_triple_per_relation))
symmetric_triple_per_relation[1] = symmetric_triple_per_relation[1].astype(str).astype(int)
symmetric_triple_per_relation_sorted = symmetric_triple_per_relation.sort_values(by=1, ascending=False)

# in the test set:
# In the test set
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
#############################################################################################################
# original_array = np.array(original_list)
# original_hrt_triple_for_symmetric = pd.DataFrame(np.array(original_array))
# symmetric_array = np.array(symmetric_list)
# symmetric_array_relation_list = pd.DataFrame(symmetric_array)[1]
# symmetric_array_df = pd.DataFrame(symmetric_array)
# unique_relations_symmetric_triples = np.array(list(set(list(symmetric_array_df[1]))))
#
#
# count =0
# symmetric_triple_per_relation = []
# #Check how many hrt and trh exist for the above mentioned symmetric relation r
# for unique_relations_symmetric_triple in unique_relations_symmetric_triples:
#     #print(count)
#     symmetric_array_place_holder = symmetric_array_df.loc[symmetric_array_df[1]==unique_relations_symmetric_triple]
#     symmetric_array_place_holder = np.array(symmetric_array_place_holder)
#     # print(unique_relations_reflexive_triple, ' : ', len(place_holder_relation_triple))
#     stat_count_per_relation_symmetric = np.array([unique_relations_symmetric_triple, int(len(symmetric_array_place_holder))])
#     symmetric_triple_per_relation.append(stat_count_per_relation_symmetric)
#
# symmetric_triple_per_relation = pd.DataFrame(np.array(symmetric_triple_per_relation))
# symmetric_triple_per_relation[1] = symmetric_triple_per_relation[1].astype(str).astype(int)
# symmetric_triple_per_relation_sorted = symmetric_triple_per_relation.sort_values(by=1, ascending=False)
#
# #In the test set
# symmetric_relation_array = np.array(symmetric_triple_per_relation_sorted[0])
#
# count_list_symmetric = np.zeros((0,3))
#
# for symmetric_relation in symmetric_relation_array:
#     place_holder_relation_test_triple = all_test_triple.loc[all_test_triple[1]==symmetric_relation]
#     place_holder_relation_test_triple_array = np.array(place_holder_relation_test_triple)
#     count_list_symmetric = np.vstack([count_list_symmetric , place_holder_relation_test_triple_array])
#
# print('total number of symmetric exist in the training set: ')
# print(len(symmetric_array_df))
# print('symmetric per relation exist in the training set: ')
# print(symmetric_triple_per_relation_sorted)
# print('for each R in the training set for which symmetric exist, how many triple exist in test set: ')
# print(len(count_list_symmetric))
#
# #Save such test triples
# #data_dir = '/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18'
# write_dir = os.path.join(data_dir,'test_symmetric.txt')
# write_to_txt_file(write_dir, count_list_symmetric)
'''
count =0
#Check how many hrt and trh exist for the above mentioned symmetric relation r
for unique_relations_symmetric_triple in unique_relations_symmetric_triples:
    #print(count)
    symmetric_array_place_holder = symmetric_array_df.loc[symmetric_array_df[1]==unique_relations_symmetric_triple]
    all_hrt = original_hrt_triple_for_symmetric.loc[original_hrt_triple_for_symmetric[1]==unique_relations_symmetric_triple]
    all_hrt_array = np.array(all_hrt)
    for an_hrt in all_hrt_array:
        trh = np.array([an_hrt[2],an_hrt[1],an_hrt[0]])
        x = all_training_triple.loc[(trh[0]==all_training_triple[0]) & (trh[1]==all_training_triple[1])&(trh[2]==all_training_triple[2])]
        print(len(x))
    count+=1
    #all_trh =
'''
#######################################For implication###################################################################
# for fb15k and fb15k237
# implication_grounding = pd.read_table('/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/FB15k-237/groundings_implication_original.txt', header=None)
# relation_set_with_implication_premise = list(set(implication_grounding[2]))
# relation_set_with_implication_conclusion = list(set(implication_grounding[2]))
# total_relation_from_grounding = relation_set_with_implication_premise + relation_set_with_implication_conclusion
# all_relations_train = list(set(total_relation_from_grounding))


all_relations_train = list(set(list(all_training_triple[1])))
# all_relations_train_combination = [comb for comb in combinations(all_relations_train, 2)]

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
    # for comb in all_relations_train_combination:
    #      try:
    #          exists = observed_triples[triple[0], triple[1], triple[2]]
    #          exists_conclusion = o
    #          exists_premise = observed_triples[triple[0], comb[0], triple[2]]
    #          exists_conclusion = observed_triples[triple[0], comb[1], triple[2]]
    #          #print('premise ', triple)
    #          premise = np.array([triple[0], comb[0], triple[2]])
    #          conclusion = np.array([triple[0], comb[1], triple[2]])
    #          print('premise ', premise)
    #          print('conclusion ', conclusion)
    #          print('######################################')
    #          premise_list.append(np.array(premise))
    #          conclusion_list.append(np.array(conclusion))
    #      except:
    #          continue

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

# in the test set:
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

# Save such test triples
# data_dir = '/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18'
write_dir = os.path.join(data_dir, 'implication.txt')
write_to_txt_file(write_dir, count_list_implication)

#################################Inverse##########################################################################
# inverse_grounding = pd.read_table('/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/FB15k/groundings_inverse_original_test.txt', header=None)
# relation_set_with_inverse_premise = list(set(inverse_grounding[2]))
# relation_set_with_inverse_conclusion = list(set(inverse_grounding[2]))
# total_relation_from_grounding = relation_set_with_inverse_premise + relation_set_with_inverse_conclusion
# all_relations_train = list(set(total_relation_from_grounding))


all_relations_train = list(set(list(all_training_triple[1])))
# all_relations_train_combination = [comb for comb in combinations(all_relations_train, 2)]

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
# for triple in training_triples:
#     for comb in all_relations_train_combination:
#          try:
#              exists_premise = observed_triples[triple[0], comb[0], triple[2]]
#              exists_conclusion = observed_triples[triple[2], comb[1], triple[0]]
#              #print('premise ', triple)
#              premise = np.array([triple[0], comb[0], triple[2]])
#              conclusion = np.array([triple[2], comb[1], triple[0]])
#              print('premise ', premise)
#              print('conclusion ', conclusion)
#              print('######################################')
#              premise_list.append(np.array(premise))
#              conclusion_list.append(np.array(conclusion))
#          except:
#              continue


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
print(len(count_list_inverse))

# Save such test triples
# data_dir = '/media/mirza/Samsung_T5/Mojtaba_new/KnowledgeGraphEmbedding-masterAdaptive/data/wn18'
write_dir = os.path.join(data_dir, 'inverse.txt')
write_to_txt_file(write_dir, count_list_inverse)

# count = 0
#
# for symmetric_array_row in symmetric_array:
#     print(count)
#     all_hrt =  all_training_triple.loc[(all_training_triple[0]==symmetric_array_row[2])&(all_training_triple[1]==symmetric_array_row[1])&(all_training_triple[2]==symmetric_array_row[0])]
#     all_trh =  all_training_triple.loc[(all_training_triple[0]==symmetric_array_row[0])&(all_training_triple[1]==symmetric_array_row[1])&(all_training_triple[2]==symmetric_array_row[2])]
#     '''
#     hrt = all_training_triple[
#             (all_training_triple[0] == symmetric_array_row[2]) & (all_training_triple[1] == symmetric_array_row[1]) & (
#                     all_training_triple[2] == symmetric_array_row[0])]
#     trh = all_training_triple[
#         (all_training_triple[0] == symmetric_array_row[0]) & (all_training_triple[1] == symmetric_array_row[1]) & (
#                 all_training_triple[2] == symmetric_array_row[2])]
#     '''
#     print(all_hrt)
#     #print('hrt: ', len(all_hrt))
#     #print('trh: ', len(all_trh))
#     count+=1
#     print('###############################')
#
#
# from itertools import combinations
#
# L = [1, 2, 3, 4]
#
# x = [ comb for comb in combinations(L, 2)]
