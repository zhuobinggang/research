# 考察：为了证明举例没有意义，对于一个case，验证同一个手法的不同模型得出的结果各异。
# from manual_exp.mld2 import datas as mld2 
# from manual_exp.mld import mld
# NOTE: 不知道为什么保存下来的mld py文件会缺字，下次直接从manual exp文件里读，不要shuffle，不然找不到对应了
from sec_paragraph import dry_run_output_posibility
from mainichi_paragraph import load_customized_loader
import torch as t

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
    return res

def run(idxs, ld):
    aux = multi_res(aux_paths, idxs, ld)
    fl = multi_res(fl_paths, idxs, ld)
    aux_fl = multi_res(aux_fl_paths, idxs, ld)
    stand = multi_res(stand_paths, idxs, ld)
    return aux_fl, aux, fl, stand
