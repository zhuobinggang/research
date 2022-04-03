# 事例考察
# from manual_exp.mld2 import datas as mld2 
# from manual_exp.mld import mld
# NOTE: 不知道为什么保存下来的mld py文件会缺字，下次直接从manual exp文件里读，不要shuffle，不然找不到对应了
from sec_paragraph import dry_run_output_posibility
from mainichi_paragraph import load_customized_loader
import torch as t
import numpy as np
import requests

# RANGE = (0, 5)
LENGTH = 10
RANGE = (0, LENGTH)
aux_fl_paths = [f'save/r01_fl50_{i}.tch' for i in range(*RANGE)]
fl_paths = [f'save/fl20_{i}_e3.tch' for i in range(*RANGE)]
aux_paths = [f'save/r02_{i}.tch' for i in range(*RANGE)]
stand_paths = [f'save/stand_{i}.tch' for i in range(*RANGE)]

def request_my_logger(dic, desc = 'No describe'):
  try:
    url = "https://hookb.in/jeP3mJJdQzF9dlMMmJ7B"
    dic['desc'] = desc
    requests.post(url, json=dic)
  except:
    print('Something went wrong in request_my_logger()')



def load_mld(path = 'manual_exp'):
    return load_customized_loader(file_name = path, half = 2, batch = 1, shuffle = False, mini = False)

def beuty_output(m, idx, ld):
    out, tar = dry_run_output_posibility(m, ld[idx])
    return round(out.item(), 4)

def get_all_target_one_idxs(ld):
    res = []
    for idx, case in enumerate(ld):
        ss, ls, pos = case[0]
        if (pos != 0 and ls[pos] == 1):
            res.append(idx)
    return res

def get_all_target_zero_idxs(ld):
    res = []
    for idx, case in enumerate(ld):
        ss, ls, pos = case[0]
        if (pos != 0 and ls[pos] == 0):
            res.append(idx)
    return res


# return: (LENGTH, n)
def multi_res(paths, idxs, ld):
    res = []
    for path in paths:
        m = t.load(path)
        m_cases = [beuty_output(m, idx, ld) for idx in idxs]
        res.append(m_cases)
    return np.array(res)

# return: (4, LENGTH, n)
def method4_model5_res(idxs, ld):
    aux_fl = multi_res(aux_fl_paths, idxs, ld)
    aux = multi_res(aux_paths, idxs, ld)
    fl = multi_res(fl_paths, idxs, ld)
    stand = multi_res(stand_paths, idxs, ld)
    return np.array([aux_fl, aux, fl, stand])

def best_idx(model_mean_outputs, focus_idx, MAX = True):
    assert len(model_mean_outputs) == 4 
    assert model_mean_outputs[0].shape[0] == LENGTH # (4, 5, n)
    model_mean_outputs = [res.mean(0) for res in model_mean_outputs] # (4, n)
    model_mean_outputs = np.array(model_mean_outputs) # (4, n)
    # assert model_mean_outputs.shape == (4, 100)
    model_mean_outputs = model_mean_outputs.transpose() # (n, 4)
    results = []
    for local_idx, row in enumerate(model_mean_outputs):
        if focus_idx == -1: # 将所有例子都拿出来
            results.append((local_idx,row))
        else:
            if MAX:
                if row.argmax() == focus_idx:
                    results.append((local_idx,row))
            else:
                if row.argmin() == focus_idx:
                    results.append((local_idx,row))
            # if MAX and row.argmax() == focus_idx:
            #     results.append((local_idx,row))
            # elif not MAX and :
            #     results.append((local_idx,row))
    return results

# return: (global_idxs, case, methods_mean_posibility)
def best_focus(model_mean_outputs, max_idx, ld, org_idxs, MAX = True):
    local_idx_and_row = best_idx(model_mean_outputs, max_idx, MAX)
    return [(org_idxs[local_idx], ld[org_idxs[local_idx]], row) for local_idx, row in local_idx_and_row]

def run():
    ld = load_mld()
    # org_idxs = get_all_target_one_idxs(ld)
    org_idxs = get_all_target_zero_idxs(ld)
    dd = method4_model5_res(org_idxs, ld) # (4, LENGTH, n)
    r0 = best_focus(dd, 0, ld, org_idxs, MAX = False)
    r1 = best_focus(dd, 1, ld, org_idxs, MAX = False)
    r2 = best_focus(dd, 2, ld, org_idxs, MAX = False)
    r3 = best_focus(dd, 3, ld, org_idxs, MAX = False)
    return [r0, r1, r2, r3], dd


def get_dd0_dd1(org_idxs_0, org_idxs_1, ld):
    dd0 = method4_model5_res(org_idxs_0, ld) # (4, len(org_idxs_0), n)
    dd1 = method4_model5_res(org_idxs_1, ld) # (4, len(org_idxs_1), n)
    return dd0, dd1



def filter(r, idx):
    res = []
    for item in r:
        _, _, out = item
        if out[idx] > 0.5:
            res.append(item)
    return res

# NOTE： 使用时注意更改逻辑
def filter_aux_loss_win(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] < 0.5 and row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        # if row[0] > 0.5 and row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]
        

def filter_focal_loss_win(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] < 0.5 and row[2] < 0.5 and row[1] > 0.5 and row[3] > 0.5]
        # if row[0] > 0.5 and row[2] > 0.5 and row[1] < 0.5 and row[3] < 0.5]

def filter_vanilla_win(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[3] < 0.5 and row[1] > 0.5 and row[2] > 0.5 and row[0] > 0.5]


def filter_aux_loss_loss(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        # if row[0] < 0.5 and row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        if row[0] > 0.5 and row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_fl_loss_loss(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] > 0.5 and row[2] > 0.5 and row[1] < 0.5 and row[3] < 0.5]
        # if row[0] < 0.5 and row[2] < 0.5 and row[1] > 0.5 and row[3] > 0.5]

def filter_vanilla_loss(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[3] > 0.5 and row[0] < 0.5 and row[1] < 0.5 and row[2] < 0.5]


# NOTE: 下面的几个方法是除开aux_fl只比较剩余三个

def filter_aux_only_win_1(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        # if row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        if row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_aux_only_win_0(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        # if row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_fl_only_win_1(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        # if row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        if row[2] > 0.5 and row[1] < 0.5 and row[3] < 0.5]

def filter_fl_only_win_0(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[2] < 0.5 and row[1] > 0.5 and row[3] > 0.5]
        # if row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_aux_only_loss_1(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        # if row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_aux_only_loss_0(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        # if row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        if row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_fl_only_loss_1(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[2] < 0.5 and row[1] > 0.5 and row[3] > 0.5]
        # if row[2] > 0.5 and row[1] < 0.5 and row[3] < 0.5]

def filter_fl_only_loss_0(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        # if row[2] < 0.5 and row[1] > 0.5 and row[3] > 0.5]
        if row[2] > 0.5 and row[1] < 0.5 and row[3] < 0.5]

# NOTE： 在EXP7上增加考察，计算使用aux之后胜出&败北的例子百分比
def run_aux_win_and_loss(path):
    ld = load_mld(path)
    org_idxs_1 = get_all_target_one_idxs(ld)
    org_idxs_0 = get_all_target_zero_idxs(ld)
    dd1 = method4_model5_res(org_idxs_1, ld) # (4, len(org_idxs_1), n)
    dd0 = method4_model5_res(org_idxs_0, ld) # (4, len(org_idxs_0), n)
    r1 = best_focus(dd1, -1, ld, org_idxs_1)
    r0 = best_focus(dd0, -1, ld, org_idxs_0)
    dic = {}
    dic['auxwin1'] = len(filter_aux_only_win_1(r1))
    dic['auxwin0'] = len(filter_aux_only_win_0(r0))
    dic['flwin1'] = len(filter_fl_only_win_1(r1))
    dic['flwin0'] = len(filter_fl_only_win_0(r0))
    dic['auxlose1'] = len(filter_aux_only_loss_1(r1))
    dic['auxlose0'] = len(filter_aux_only_loss_0(r0))
    dic['fllose1'] = len(filter_fl_only_loss_1(r1))
    dic['fllose0'] = len(filter_fl_only_loss_0(r0))
    request_my_logger(dic, desc = f"{len(org_idxs_1)}, {len(org_idxs_0)}, 用计算器看看是否跟之前的百分比相近")
    return r0, r1

def run2():
    r10, r11 = run_aux_win_and_loss('test0')
    r00, r01 = run_aux_win_and_loss('manual_exp')
    return r00, r01, r10, r11
