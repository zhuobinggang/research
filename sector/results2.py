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

standard_cls_2vs2_early = [ 
  {'prec': 0.7576866764275256, 'rec': 0.6456643792888335, 'f1': 0.6972044459413944, 'bacc': 0.774069667546597, 'index': 1, 'valid_loss': 2013.1536898463964}, 
  {'prec': 0.7081545064377682, 'rec': 0.7205240174672489, 'f1': 0.7142857142857142, 'bacc': 0.7901382609434064, 'index': 1, 'valid_loss': 1974.137339234352},
  {'prec': 0.7400673400673401, 'rec': 0.6855895196506551, 'f1': 0.7117875647668396, 'bacc': 0.7859297038441844, 'index': 1, 'valid_loss': 2088.899380683899},
]

sector_2vs2_early_rate02 = [
  {'prec': 0.7491090520313614, 'rec': 0.6556456643792888, 'f1': 0.699268130405855, 'bacc': 0.775966615336374, 'index': 1, 'valid_loss': 2831.0838955938816}, 
  {'prec': 0.764367816091954, 'rec': 0.6637554585152838, 'f1': 0.7105175292153589, 'bacc': 0.7835571635534581, 'index': 1, 'valid_loss': 2789.694347858429}, 
  {'prec': 0.7475795297372061, 'rec': 0.6743605739238927, 'f1': 0.7090849458838964, 'bacc': 0.783408925736254, 'index': 1, 'valid_loss': 2940.132279574871},
]

