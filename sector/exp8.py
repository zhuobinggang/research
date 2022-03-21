# 选出最受瞩目的token，用wordcloud来展示
from sec_paragraph import get_att_ours
from exp7 import load_mld
import torch as t

def load_m():
    return t.load('save/r01_fl50_0.tch')
    # return t.load('save/analyse_r02_para.tch')

# 从mld选出取出正确率最高的100个例子
def get_info_ranking_by_posibility():
    m = load_m()
    pcases = cases_positive(load_mld())
    need = cases_ranking_from_max(m, pcases)
    return need, m.toker
    

def cases_positive(ld):
    results = []
    for case in ld:
        ss, labels, pos = case[0]
        if labels[pos] == 1:
            results.append(case)
    return results

def cases_ranking_from_max(m, cases):
    need = []
    for case in cases:
        atts, idss, results, targets, labelss = get_att_ours(m, case)
        need.append((results[0], case, atts[0], idss[0]))
    need = list(reversed(sorted(need, key = lambda x : x[0])))
    return need

