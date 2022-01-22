# 考察：为了证明举例没有意义，对于一个case，验证同一个手法的不同模型得出的结果各异。
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


def multi_res(paths, idxs, ld):
    res = []
    for path in paths:
        m = t.load(path)
        m_cases = [beuty_output(m, idx, ld) for idx in idxs]
        res.append(m_cases)
    return np.array(res)

def cal_mean_res(idxs, ld):
    aux_fl = multi_res(aux_fl_paths, idxs, ld)
    aux = multi_res(aux_paths, idxs, ld)
    fl = multi_res(fl_paths, idxs, ld)
    stand = multi_res(stand_paths, idxs, ld)
    return aux_fl, aux, fl, stand

def best_idx(ress, max_idx):
    assert len(ress) == 4 
    assert ress[0].shape == (5, 100)
    ress = [res.mean(0) for res in ress]
    ress = np.array(ress)
    assert ress.shape == (4, 100)
    ress = ress.transpose()
    results = []
    for idx, col in enumerate(ress):
        if col.argmax() == max_idx:
            results.append((idx,col))
    return results

def best_idx_map_to_org_idx(ress, max_idx, ld, org_idxs):
    best_idx_out_in_range = best_idx(ress, max_idx, ld)
    return [(org_idxs[idx], ld[org_idxs[idx]], out) for idx, out in best_idx_out_in_range]

def run():
    ld = load_mld()
    org_idxs = get_all_target_one_idxs(ld)[:100]
    aux_fl, aux, fl, stand = cal_mean_res(org_idxs, ld)
    r0 = best_idx_map_to_org_idx([aux_fl, aux, fl, stand], 0, ld, idxs)
    r1 = best_idx_map_to_org_idx([aux_fl, aux, fl, stand], 1, ld, idxs)
    r2 = best_idx_map_to_org_idx([aux_fl, aux, fl, stand], 2, ld, idxs)
    r3 = best_idx_map_to_org_idx([aux_fl, aux, fl, stand], 3, ld, idxs)
    return [r0, r1, r2, r3], [aux_fl, aux, fl, stand]
    

