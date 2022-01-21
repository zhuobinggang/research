# 考察：为了证明举例没有意义，对于一个case，验证同一个手法的不同模型得出的结果各异。

from manual_exp.mld import mld
from sec_paragraph import dry_run_output_posibility
import torch as t

def beuty_output(m, idx):
    out, tar = dry_run_output_posibility(m, mld[idx])
    return round(out.item(), 4), round(tar.item(), 4)


