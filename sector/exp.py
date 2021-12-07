# 为了“考察”部分做实验
import mainichi_paragraph as custom_data
# from manual_exp.mld import mld
from sec_paragraph import *

def run():
    manual_exp_ld = custom_data.load_customized_loader(file_name='manual_exp', half=2, batch=1, mini=False, shuffle = True)
    mld = manual_exp_ld
    mld = ld_without_opening(mld)
    return mld

def ld_without_opening(ld):
    ld = [case for case in ld if case[0][2] != 0]
    return ld


# 让机器来跑特定，取40个平均值
def get_average(mld, length = 40):
    res_fl_aux = []
    res_fl = []
    res_bce_aux = []
    res_bce = []
    for i in range(length):
        res_fl_aux.append(get_test_result_dic(t.load(f'save/r01_fl50_{i}.tch'), mld))
        res_fl.append(get_test_result_dic(t.load(f'save/fl20_{i}.tch'), mld))
        res_bce.append(get_test_result_dic(t.load(f'save/stand_{i}.tch'), mld))
        res_bce_aux.append(get_test_result_dic(t.load(f'save/r01_{i}.tch'), mld))
    return res_fl_aux, res_fl, res_bce_aux, res_bce


def get_n_average(test = False):
    mld = run()
    if test:
        mlds = [mld[:2]]
        length = 1
    else:
        mlds = [mld[:100], mld[100:200], mld[200:300], mld[300:400], mld[400:500]]
        length = 40
    res_fl_auxs = []
    res_fls = []
    res_bce_auxs = []
    res_bces = []
    for mld in mlds:
        res_fl_aux, res_fl, res_bce_aux, res_bce = get_average(mld, length)
        res_fl_auxs.append(res_fl_aux)
        res_fls.append(res_fl)
        res_bce_auxs.append(res_bce_aux)
        res_bces.append(res_bce)
    return res_fl_auxs, res_fls, res_bce_auxs, res_bces


