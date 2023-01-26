# 人手实验
# from sec_paragraph import *
from exp_news import *
# from manual_exp.mld2 import datas as mld
from manual_exp.mld import mld
import numpy as np
import json

def transfer_mld(mld):
    res = []
    for item in mld:
        ss, labels, pos = item[0]
        if len(labels) < 4:
            if pos == 1: 
                labels = [None] + labels
                ss = [None] + ss
            elif pos == 2:
                labels = labels + [None]
                ss = ss + [None]
            else:
                print('ERR!')
        res.append((ss, labels))
    return res

def run():
    ld = transfer_mld(mld)
    times = 10
    res = []
    for model_idx_org in range(times):
        SEED = SEEDS_FOR_TRAIN[model_idx_org]
        m = load_model(f'SEED{SEED}_AUX01FL50E2')
        res.append(test_chain(m, ld))
    return res


        
def save_to_json(mld):
    text = json.dumps(mld)
    f = open('novel_mld.json', 'w')
    f.write(text)
    f.close()
    return text







