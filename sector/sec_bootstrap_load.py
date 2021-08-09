import random
import utils as U
import bootstrap_result
import numpy as np
result_dic = bootstrap_result.results
targets = result_dic['targets']
multi_stands = result_dic['outputs_stand']
multi_mys = result_dic['outputs_my']
multi_fls = result_dic['outputs_FL']

def generate_sampled_multi_indexs(length, time = 1000):
    sampled_multi_indexs = []
    for i in range(time):
        temp_idxs = []
        for i in range(length):
            temp_idxs.append(random.randint(0, length - 1))
        sampled_multi_indexs.append(temp_idxs)
    return sampled_multi_indexs

def cal_avg_f_score(targets, multi_results, sampled_indexs):
    sampled_targets = [targets[idx] for idx in sampled_indexs]
    sampled_multi_results = []
    for results in multi_results:
        sampled_multi_results.append([results[idx] for idx in sampled_indexs])
    f_scores = []
    for sampled_results in sampled_multi_results:
        prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(sampled_results, sampled_targets)
        f_scores.append(f1)
    return np.average(f_scores)

def cal_bootstrap_f_score(targets, multi_results):
    sampled_indexs = []
    for i in range(len(targets)):
        sampled_indexs.append(random.randint(0, len(targets) - 1))
    return cal_avg_f_score(targets, multi_results, sampled_indexs)


def cal_huge_times(time = 1000):
    sampled_multi_indexs = generate_sampled_multi_indexs(len(targets), time)
    avg_fs_stand = []
    avg_fs_mys = []
    avg_fs_fls = []
    for sampled_indexs in sampled_multi_indexs:
        avg_fs_stand.append(cal_avg_f_score(targets, multi_stands, sampled_indexs))
        avg_fs_mys.append(cal_avg_f_score(targets, multi_mys, sampled_indexs))
        avg_fs_fls.append(cal_avg_f_score(targets, multi_fls, sampled_indexs))
    return avg_fs_stand, avg_fs_mys, avg_fs_fls


def cal_win_rate(fs1, fs2):
    total_cnt = len(fs1)
    win_cnt = 0
    for f1, f2 in zip(fs1, fs2):
        win_cnt += 1 if f1 > f2 else 0
    return win_cnt / total_cnt

# Example
def run():
    avg_fs_stand, avg_fs_mys, avg_fs_fls = cal_huge_times(10000)
    print(f'My vs Stand, win rate: {cal_win_rate(avg_fs_mys, avg_fs_stand)}')
    print(f'My vs Fls, win rate: {cal_win_rate(avg_fs_mys, avg_fs_fls)}')
    return avg_fs_stand, avg_fs_mys, avg_fs_fls



