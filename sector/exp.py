# 为了“考察”部分做实验
import mainichi_paragraph as custom_data
# from manual_exp.mld import mld
from sec_paragraph import *

def get_mld():
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


def split_mlds(mld, n_hundreds = 5):
    mlds = []
    for i in range(n_hundreds):
        mlds.append(mld[ i*100 : i*100 + 100 ])
        # mlds.append((i*100, i*100 + 100))
    return mlds


def get_n_average(mld, n_hundreds = 5, test = False):
    # mld = run()
    if test:
        mlds = [mld[:2]]
        length = 1
    else:
        mlds = split_mlds(mld, n_hundreds)
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


def ress_to_avg_dic(ress):
    f1 = np.average([item['f1'] for item in ress])
    prec = np.average([item['prec'] for item in ress])
    rec = np.average([item['rec'] for item in ress])
    return {'f1': f1, 'prec': prec, 'rec': rec}

def resss_to_avg_dic(resss):
    dics = [ress_to_avg_dic(item) for item in resss]
    f1 = np.average([dic['f1'] for dic in dics])
    prec = np.average([dic['prec'] for dic in dics])
    rec = np.average([dic['rec'] for dic in dics])
    return {'f1': f1, 'prec': prec, 'rec': rec}


# 第一次实验: 5个数据集(和最好的只差了0.001)
# 
#  >>> resss_to_avg_dic(res_fl_auxs)
#  {'f1': 0.7127853633064213, 'prec': 0.7359941484337351, 'rec': 0.698188084464555}
#  >>> resss_to_avg_dic(res_fls)
#  {'f1': 0.7131974415707419, 'prec': 0.7413405506011619, 'rec': 0.6964834087481145}
#  >>> resss_to_avg_dic(res_bces)
#  {'f1': 0.7088121790491017, 'prec': 0.7477118003665765, 'rec': 0.6846093514328808}
#  >>> resss_to_avg_dic(res_bce_auxs)
#  {'f1': 0.7046932948502371, 'prec': 0.7509483445729673, 'rec': 0.672867269984917}

G = {ress: None}

def step1()
    mld = get_mld()
    res_fl_auxs, res_fls, res_bce_auxs, res_bces = get_n_average(mld, 10)
    G['ress'] = (res_fl_auxs, res_fls, res_bce_auxs, res_bces)
    print('First step over, start step2')

