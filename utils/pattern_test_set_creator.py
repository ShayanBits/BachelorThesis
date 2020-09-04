import pandas as pd
import os


def write_to_txt_file(path, data):
    f = open(path, "w")
    data.to_string(f, col_space=1, index=False,  header=False)
    f.close()

dirname = os.path.dirname(__file__)
dataset = "wn18rr"
# pattern = "implication"
patterns = ["one_to_many", "implication", "inverse", "symmetric"]
relPathToData = "../data/"
N = 3

data_dir = os.path.join(dirname, relPathToData, dataset)

# Load excel file of the pattern

pathToTestTriples = os.path.join(data_dir, "test.txt")
all_test_triple = pd.read_table(pathToTestTriples, header=None)

all_relations_test = list(set(list(all_test_triple[1])))

for pattern in patterns:
    pathToExcel = os.path.join(data_dir, pattern + "_stats_train.xlsx")
    patternResultsTable = pd.read_excel(pathToExcel, sheet_name=0)

    # check if pattern is one_to_many or not, sort columns and take the first N relations
    if pattern == "one_to_many":
        patternResultsTable = patternResultsTable.sort_values('avg', ascending=False)
    else:
        patternResultsTable = patternResultsTable.sort_values('count', ascending=False)

    top_n_relations = patternResultsTable.iloc[0:3, 0]
    top_n_relations = list(top_n_relations.values.flatten())
    new_test_triples = pd.DataFrame()
    for relation in top_n_relations:
        temp = all_test_triple[all_test_triple[1] == relation]
        new_test_triples = new_test_triples.append(temp, ignore_index=True)

    write_dir = os.path.join(data_dir, pattern + '_test.txt')
    write_to_txt_file(write_dir, new_test_triples)

