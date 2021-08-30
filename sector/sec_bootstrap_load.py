import random
import utils as U
import g_save_2v2
import numpy as np
result_dic = g_save_2v2.result
targets = result_dic['targets']
multi_stands = result_dic['outputs_stand']
multi_mys = result_dic['outputs_my']
multi_fls = result_dic['outputs_FL']
multi_all_one = [[1] * 6635]

def generate_sampled_multi_indexs(length, time = 1000):
    sampled_multi_indexs = []
    for i in range(time):
        temp_idxs = []
        for i in range(length):
            temp_idxs.append(random.randint(0, length - 1))
        sampled_multi_indexs.append(temp_idxs)
    return sampled_multi_indexs

def cal_avg_score(targets, multi_results, sampled_indexs):
    sampled_targets = [targets[idx] for idx in sampled_indexs]
    sampled_multi_results = []
    for results in multi_results:
        sampled_multi_results.append([results[idx] for idx in sampled_indexs])
    f_scores = []
    precs = []
    recs = []
    for sampled_results in sampled_multi_results:
        prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(sampled_results, sampled_targets)
        f_scores.append(f1)
        precs.append(prec)
        recs.append(rec)
    return np.average(f_scores), np.average(precs), np.average(recs)

def cal_bootstrap_f_score(targets, multi_results):
    sampled_indexs = []
    for i in range(len(targets)):
        sampled_indexs.append(random.randint(0, len(targets) - 1))
    return cal_avg_score(targets, multi_results, sampled_indexs)[0]


def cal_huge_times(time = 1000):
    sampled_multi_indexs = generate_sampled_multi_indexs(len(targets), time)
    avg_all_stand = []
    avg_all_mys = []
    avg_all_fls = []
    avg_all_one = []
    for sampled_indexs in sampled_multi_indexs:
        avg_all_stand.append(cal_avg_score(targets, multi_stands, sampled_indexs))
        avg_all_mys.append(cal_avg_score(targets, multi_mys, sampled_indexs))
        avg_all_fls.append(cal_avg_score(targets, multi_fls, sampled_indexs))
        avg_all_one.append(cal_avg_score(targets, multi_all_one, sampled_indexs))
    return avg_all_stand, avg_all_mys, avg_all_fls, avg_all_one


def cal_win_rate(fs1, fs2):
    total_cnt = len(fs1)
    win_cnt = 0
    for f1, f2 in zip(fs1, fs2):
        win_cnt += 1 if f1 > f2 else 0
    return win_cnt / total_cnt

# Example
def run():
    avg_all_stand, avg_all_mys, avg_all_fls, avg_all_one = cal_huge_times(10000)
    avg_fs_stand, avg_fs_mys, avg_fs_fls, avg_fs_all_one = [item[0] for item in avg_all_stand], [item[0] for item in avg_all_mys], [item[0] for item in avg_all_fls], [item[0] for item in avg_all_one]
    print(f'My vs Stand, win rate: {cal_win_rate(avg_fs_mys, avg_fs_stand)}')
    print(f'My vs Fls, win rate: {cal_win_rate(avg_fs_mys, avg_fs_fls)}')
    return avg_fs_stand, avg_fs_mys, avg_fs_fls, avg_fs_all_one



