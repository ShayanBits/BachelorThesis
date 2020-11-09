# Copyright © 2020 Shayan Shahpasand

# ===============================================================
# ==      ===  ====  =====  =====  ====  =====  =====  =======  =
# =  ====  ==  ====  ====    ====   ==   ====    ====   ======  =
# =  ====  ==  ====  ===  ==  ====  ==  ====  ==  ===    =====  =
# ==  =======  ====  ==  ====  ===  ==  ===  ====  ==  ==  ===  =
# ====  =====        ==  ====  ====    ====  ====  ==  ===  ==  =
# ======  ===  ====  ==        =====  =====        ==  ====  =  =
# =  ====  ==  ====  ==  ====  =====  =====  ====  ==  =====    =
# =  ====  ==  ====  ==  ====  =====  =====  ====  ==  ======   =
# ==      ===  ====  ==  ====  =====  =====  ====  ==  =======  =
# ===============================================================

# This log to excel converter is built by me during my bachelor thesis
# at university of Bonn. If you use it as a whole or part, I would appreciate
# if you acknowledge my work by mention it. This code is published under MIT license.
# shayan.shahpasand@uni-bonn.de


import os
import pandas as pd

field_to_check = "HITS@1"

dataset = "fb15k"
dataset_directory_name = "FB15k"
patterns = ["symmetric", "inverse", "implication", "one_to_many"]
# patterns = ["default", "symmetric", "inverse", "implication", "one_to_many"]
pattern = "one_to_many"

code_path = "../codes"
data_path = f"../data/{dataset_directory_name}"

absolute_path_to_excel = "/Volumes/SHAYAN/BT/results/"

relPathToExcel = os.path.join(absolute_path_to_excel + dataset + "/excel/", "results-" + dataset + "-")
exactPathToExcel = os.path.join(relPathToExcel + pattern + ".xlsx")
currentSheet = pd.read_excel(exactPathToExcel, sheet_name=0)
# finding rows without result
rows_without_result = currentSheet.loc[currentSheet[field_to_check].isnull()]

commands = []
for index, row in rows_without_result.iterrows():
    model = row['Model']
    hidden_dimension = row['hidden dimension']
    negs = row['negative sample size']
    batch_size = row['batch size']
    gamma = int(row['gamma'])
    temperature = int(row['adversarial_temperature'])
    learning_rate = row['learning rate']
    loss = row['loss']

    SAVE_PATH = f"../models/{model}/{loss}/{dataset_directory_name}"
    COMPLETE_SAVE_PATH = f"{SAVE_PATH}/dim-{hidden_dimension}/gamma-{gamma}/learning-rate-{learning_rate}/batch-size-{batch_size}/negative-sample-size-{negs}/"

    # command = f"python3 {code_path}/run.py --do_grid --cuda --do_test --data_path {data_path} --model {model} -d {hidden_dimension} --negative_sample_size {negs} --batch_size {batch_size} --gamma {gamma} --adversarial_temperature {temperature} --negative_adversarial_sampling -lr {learning_rate} --max_steps 400000 -save {COMPLETE_SAVE_PATH} -de --loss {loss} --init_checkpoint {COMPLETE_SAVE_PATH} \n"
    command = f"python3 {code_path}/run.py --do_grid --cuda --do_test --data_path {data_path} --model {model} -d {hidden_dimension} --negative_sample_size {negs} --batch_size {batch_size} --gamma {gamma} --adversarial_temperature {temperature} --negative_adversarial_sampling -lr {learning_rate} --max_steps 400000 -save {COMPLETE_SAVE_PATH} -de --loss {loss} \n"
    commands.append(command)

f = open("commands.txt", "w")
f.writelines(commands)
