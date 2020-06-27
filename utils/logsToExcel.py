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
import glob
import pandas as pd
import re

extract_information = ["file name", "Model", "Data Path", "#entity", "#relation", "#train",
                       "#valid", "opt", "#test", "batch size","learning rate", "gamma",
                       "hidden dimension", "negative sample size", "adversarial_temperature",
                       "loss", "MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]

special_information = ["MRR", "MR", "HITS@1", "HITS@3", "HITS@10"]

dirname = os.path.dirname(__file__)
relPathToLogs = "../Results/RotatE/margin_ranking/FB15k/"

pathToLogs = os.path.join(dirname, relPathToLogs, '*.log')

pathToExcel = os.path.join(dirname, "../results/results.xlsx")

currentSheet = pd.read_excel(pathToExcel, sheet_name=0)


def saveResultAsExcel(dataFrame):
    # writer = pd.ExcelWriter(pathToExcel, engine='xlsxwriter')
    dataFrame.to_excel(pathToExcel, index=False)


def appendRow(keyValue):
    completeRow = dict.fromkeys(extract_information)
    for key in completeRow:
        if key in keyValue.keys():
            completeRow[key] = keyValue[key]
        else:
            completeRow[key] = ""

    newRow = pd.DataFrame([completeRow.values()], columns=completeRow.keys())
    updatedSheet = currentSheet.append(newRow, ignore_index=True)
    return updatedSheet


def updateExcelFile(pathToLogs):
    for filename in glob.glob(os.path.join(pathToLogs)):
        with open(os.path.join(filename), 'r') as logFile:
            keyValue = {"file name": os.path.basename(logFile.name)}
            logFile = logFile.readlines()
            for line in logFile:
                for phrase in extract_information:
                    if phrase in line:
                        if phrase in special_information:
                            value = re.search(r": \d*\.?\d+", line)
                            if value is not None:
                                keyValue[phrase] = value.group()[2:]
                        else:
                            value = re.search(rf"{phrase}: ?.*", line)
                            if value is not None:
                                keyValue[phrase] = value.group()[phrase.__len__() + 2:]

            updatedDataFreame = appendRow(keyValue)
            saveResultAsExcel(updatedDataFreame)


updateExcelFile(pathToLogs)
