from mainichi_results import *

def analyse(datas, dicname='testdic'):
    precs = []
    recs = []
    f1s = []
    baccs = []
    for data in datas:
        precs.append(data[dicname]['prec'])
        recs.append(data[dicname]['rec'])
        f1s.append(data[dicname]['f1'])
        baccs.append(data[dicname]['bacc'])
    return precs, recs, f1s, baccs


def get_max_fs_by_2list(raw1, raw2):
    _, _, f1, _ = analyse(raw1)
    _, _, f2, _ = analyse(raw2)
    return [max(a, b) for a, b in zip(f1, f2)]


def get_max_fs_by_mess_list(raw1):
    _, _, f1_f2, _ = analyse(raw1)
    return [max(a, b) for a, b in zip(f1_f2[0::2], f1_f2[1::2])]

def get_max_fs_by_mess3(raw1):
    _, _, f123, _ = analyse(raw1)
    return [max(a, b, c) for a, b, c in zip(f123[0::3], f123[1::3], f123[2::3])]

# ============= 辅助函数 ================


