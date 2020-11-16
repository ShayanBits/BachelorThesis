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
import numpy as np

data_dir= "../data/fb15k"

with open(os.path.join(data_dir, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join(data_dir, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)


absolute_path_to_results = "/Volumes/SHAYAN/BT/results/wn18rr/models/DistMult/adaptive_margin/wn18rr/dim-10/gamma-1/learning-rate-0.1/batch-size-512/negative-sample-size-10"
entities = np.load(os.path.join(absolute_path_to_results, 'entity_embedding0.npy'))
relations = np.load(os.path.join(absolute_path_to_results, 'relation_embedding0.npy'))

bla = "test"
# checkpoint = torch.load(os.path.join(args.init_checkpoint, 'entity_embedding0.npy' ), map_location= 'cpu')

# np.load() entity
# pick the most frequent relation from the excel file
# store the relation id from dictionary
# read entities for one config



