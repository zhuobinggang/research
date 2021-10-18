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

def get_results_2x20():
    xs = ['r00+fl10', 'r00+fl20', 'r00+fl05', 'r01+fl10','r01+fl20','r01+fl05','r02+fl10','r02+fl20', 'r02+fl05', 'stand', 'fl10', 'fl20', 'fl05', 'r00', 'r01', 'r02']
    ys_dev_pc, ys_test_pc = split_from_res(res_pc, 16)
    ys_dev_pt, ys_test_pt = split_from_res(res_pt, 16)
    legends = ['dev_1','test_1', 'dev_2', 'test_2']
    return xs, [ys_dev_pc, ys_test_pc, ys_dev_pt, ys_test_pt], legends

def get_results():
    xs = ['r00+fl10', 'r00+fl20', 'r00+fl05', 'r01+fl10','r01+fl20','r01+fl05','r02+fl10','r02+fl20', 'r02+fl05', 'stand', 'fl10', 'fl20', 'fl05', 'r00', 'r01', 'r02']
    ys_dev_pc, ys_test_pc = split_from_res(res_pc, 16)
    ys_dev_pt, ys_test_pt = split_from_res(res_pt, 16)
    legends = ['dev_1','test_1', 'dev_2', 'test_2']
    ys_dev = np.array([ys_dev_pc, ys_dev_pt]).mean(0)
    ys_test = np.array([ys_test_pc, ys_test_pt]).mean(0)
    return xs, [ys_dev, ys_test], legends
