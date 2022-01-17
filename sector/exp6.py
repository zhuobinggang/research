# cross validation: fl2.0 + e2 vs aux0.1 + fl5.0 + e2
from sec_paragraph import *
from mainichi_paragraph import read_additional_test_ds

cal_f1 = cal_f1

# return: len(tlds) fs
def cal_f1s_one_model_vs_tlds(m, tlds):
    res = []
    for tld in tlds:
        res.append(cal_f1(m, tld))
    return res

def write_to(path, thing):
    f = open(path, 'w')
    f.write(str(thing))
    f.close()

# 2 * 4 * 10个f值就可以了
def yes():
    tlds = read_additional_test_ds()
    # fl2.0 + e2
    # load 10 models for 4 times
    fsss_fl = [] # 4 * 10 * 10
    for i in range(4):
        start = i * 10
        end = i * 10 + 10
        fss = [] # 10 * 10
        for model_index in range(start, end):
            fss.append(cal_f1s_one_model_vs_tlds(t.load(f'save/fl20_{model_index}.tch')))
        fsss_fl.append(fss)
        write_to('fsss_fl.txt', fsss_fl)
    # aux0.1 + fl5.0 + e2
    # load 10 models for 4 times
    fsss_my = []
    for i in range(4):
        start = i * 10
        end = i * 10 + 10
        fss = [] # 10 * 10
        for model_index in range(start, end):
            fss.append(cal_f1s_one_model_vs_tlds(t.load(f'save/r01_fl50_{model_index}.tch')))
        fsss_my.append(fss)
        write_to('fsss_my.txt', fsss_my)

