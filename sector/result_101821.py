import numpy as np

from mess_pc_2 import *
from mess_pt_2 import *

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

def get_results_2x20():
    xs = ['fl30', 'fl40', 'fl50', 'r00+fl30','r00+fl40','r00+fl50','r01+fl30','r01+fl40', 'r01+fl50', 'r02+fl30', 'r02+fl40', 'r02+fl50']
    ys_dev_pc, ys_test_pc = split_from_res(res_pc, 12)
    ys_dev_pt, ys_test_pt = split_from_res(res_pt, 12)
    legends = ['dev_1','test_1', 'dev_2', 'test_2']
    return xs, [ys_dev_pc, ys_test_pc, ys_dev_pt, ys_test_pt], legends

def get_results():
    xs = ['fl30', 'fl40', 'fl50', 'r00+fl30','r00+fl40','r00+fl50','r01+fl30','r01+fl40', 'r01+fl50', 'r02+fl30', 'r02+fl40', 'r02+fl50']
    ys_dev_pc, ys_test_pc = split_from_res(res_pc, 12)
    ys_dev_pt, ys_test_pt = split_from_res(res_pt, 12)
    legends = ['dev_1','test_1', 'dev_2', 'test_2']
    ys_dev = np.array([ys_dev_pc, ys_dev_pt]).mean(0)
    ys_test = np.array([ys_test_pc, ys_test_pt]).mean(0)
    return xs, [ys_dev, ys_test], legends



