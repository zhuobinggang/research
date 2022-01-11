# 人手实验
from sec_paragraph import *
from manual_exp.mld2 import datas as mld
import numpy as np

def get_res_from_mld_rapid():
    res = [[], [], [], []]
    for i in range(10):
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        res[0].append(cal_f1_and_others(t.load(f'save/r01_fl50_{i}.tch'), mld))
        res[1].append(cal_f1_and_others(t.load(f'save/fl20_{i}_e3.tch'), mld))
        res[2].append(cal_f1_and_others(t.load(f'save/stand_{i}.tch'), mld))
        res[3].append(cal_f1_and_others(t.load(f'save/r02_{i}.tch'), mld))
    f = open('get_res_from_mld_rapid.txt', 'w')
    f.write(str(res))
    f.close()
    return np.array(res)

