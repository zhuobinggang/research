import numpy as np

from mess_pc import *
from mess_pt import *

def get_mat(results, key = 'f1'):
    # print(key)
    mat = []
    for item in results:
        mat.append((item[key], item['test_result'][key]))
    mat = np.array(mat)
    return mat

def split_from_res(res, num=13, step=20):
    idx = 0
    results_dev = []
    results_test = []
    for i in range(num):
        results_dev.append(get_mat(res[idx:idx+20]).mean(0)[0])
        results_test.append(get_mat(res[idx:idx+20]).mean(0)[1])
        idx += 20
    return results_dev, results_test

def get_heatmap():
    xs = ['r00+fl10', 'r00+fl20', 'r00+fl05', 'r01+fl10','r01+fl20','r01+fl05','r02+fl10','r02+fl20', 'r02+fl05', 'stand', 'fl10', 'fl20', 'fl05', 'r00']
    ys_dev_pc, ys_test_pc = split_from_res(res_pc, 14)
    ys_dev_pt, ys_test_pt = split_from_res(res_pt, 14)
    legends = ['dev_1','test_1', 'dev_2', 'test_2']
    # 由于实验出错，其中的一次20次实验没有做到，所以将20分为2个10取平均值
    ys_dev_pc[2] = 0.61182562
    ys_test_pc[2] = 0.62087113
    ys_dev_pt[2] = 0.66817857
    ys_test_pt[2] = 0.68086401
    return xs, [ys_dev_pc, ys_test_pc, ys_dev_pt, ys_test_pt], legends
