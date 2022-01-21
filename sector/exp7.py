# 考察：为了证明举例没有意义，对于一个case，验证同一个手法的不同模型得出的结果各异。

from manual_exp.mld import mld
from sec_paragraph import dry_run_output_posibility
import torch as t

def beuty_output(m, idx):
    out, tar = dry_run_output_posibility(m, mld[idx])
    return round(out.item(), 4)

def get_all_target_one_idxs():
    res = []
    for idx, case in enumerate(mld):
        ss, ls, pos = case[0]
        if ls[pos] == 1:
            res.append(idx)
    return res

LENGTH = 5
aux_fl_paths = [f'save/r01_fl50_{i}.tch' for i in range(LENGTH)]
fl_paths = [f'save/fl20_{i}_e3.tch' for i in range(LENGTH)]
aux_paths = [f'save/r02_{i}.tch' for i in range(LENGTH)]
stand_paths = [f'save/stand_{i}.tch' for i in range(LENGTH)]

def multi_res(paths, idxs):
    res = []
    for path in paths:
        m = t.load(path)
        m_cases = [beuty_output(m, idx) for idx in idxs]
        res.append(m_cases)
    return res

def run():
    aux = multi_res(aux_paths, idxs)
    fl = multi_res(fl_paths, idxs)
    aux_fl = multi_res(aux_fl_paths, idxs)
    stand = multi_res(stand_paths, idxs)
    return aux_fl, aux, fl, stand
