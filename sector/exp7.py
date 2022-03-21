# 事例考察
# from manual_exp.mld2 import datas as mld2 
# from manual_exp.mld import mld
# NOTE: 不知道为什么保存下来的mld py文件会缺字，下次直接从manual exp文件里读，不要shuffle，不然找不到对应了
from sec_paragraph import dry_run_output_posibility
from mainichi_paragraph import load_customized_loader
import torch as t
import numpy as np

LENGTH = 5
aux_fl_paths = [f'save/r01_fl50_{i}.tch' for i in range(LENGTH)]
fl_paths = [f'save/fl20_{i}_e3.tch' for i in range(LENGTH)]
aux_paths = [f'save/r02_{i}.tch' for i in range(LENGTH)]
stand_paths = [f'save/stand_{i}.tch' for i in range(LENGTH)]
LOADER_PATH = 'manual_exp' 

def load_mld():
    return load_customized_loader(file_name = LOADER_PATH, half = 2, batch = 1, shuffle = False, mini = False)

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
    

def filter(r, idx):
    res = []
    for item in r:
        _, _, out = item
        if out[idx] > 0.5:
            res.append(item)
    return res

def filter_aux_loss_win(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] < 0.5 and row[1] < 0.5 and row[2] > 0.5 and row[3] > 0.5]
        

def filter_focal_loss_win(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] < 0.5 and row[2] < 0.5 and row[1] > 0.5 and row[3] > 0.5]

def filter_vanilla_win(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[3] < 0.5 and row[1] > 0.5 and row[2] > 0.5 and row[0] > 0.5]


def filter_aux_loss_loss(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] > 0.5 and row[1] > 0.5 and row[2] < 0.5 and row[3] < 0.5]

def filter_fl_loss_loss(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[0] > 0.5 and row[2] > 0.5 and row[1] < 0.5 and row[3] < 0.5]

def filter_vanilla_loss(r):
    return [(global_idxs, case, row) for global_idxs, case, row in r 
        if row[3] > 0.5 and row[0] < 0.5 and row[1] < 0.5 and row[2] < 0.5]


