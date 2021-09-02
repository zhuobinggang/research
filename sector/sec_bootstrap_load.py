import random
import utils as U
# import g_save_2v2
import g_save_2v2_dev
import numpy as np
# result_dic = g_save_2v2.result
result_dic = g_save_2v2_dev.result
targets = result_dic['targets']

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
    avg_all_myr0 = []
    for sampled_indexs in sampled_multi_indexs:
        avg_all_stand.append(cal_avg_score(targets, result_dic['outputs_stand'], sampled_indexs))
        avg_all_mys.append(cal_avg_score(targets, result_dic['outputs_my'], sampled_indexs))
        avg_all_fls.append(cal_avg_score(targets, result_dic['outputs_FL'], sampled_indexs))
        avg_all_one.append(cal_avg_score(targets, [[1] * 6635], sampled_indexs))
        avg_all_myr0.append(cal_avg_score(targets, result_dic['outputs_my_r0'], sampled_indexs))
    return avg_all_stand, avg_all_mys, avg_all_fls, avg_all_one, avg_all_myr0


def cal_win_rate(fs1, fs2):
    total_cnt = len(fs1)
    win_cnt = 0
    for f1, f2 in zip(fs1, fs2):
        win_cnt += 1 if f1 > f2 else 0
    return win_cnt / total_cnt

# Example
def run():
    avg_all_stand, avg_all_mys, avg_all_fls, avg_all_one, avg_all_myr0 = cal_huge_times(10000)
    avg_fs_stand, avg_fs_mys, avg_fs_fls, avg_fs_all_one, avg_fs_all_myr0 = [item[0] for item in avg_all_stand], [item[0] for item in avg_all_mys], [item[0] for item in avg_all_fls], [item[0] for item in avg_all_one], [item[0] for item in avg_all_myr0]
    print(f'My vs Stand, win rate: {cal_win_rate(avg_fs_mys, avg_fs_stand)}')
    print(f'My vs Fls, win rate: {cal_win_rate(avg_fs_mys, avg_fs_fls)}')
    avg_prec_stand, avg_prec_mys, avg_prec_fls, avg_prec_all_one = [item[1] for item in avg_all_stand], [item[1] for item in avg_all_mys], [item[1] for item in avg_all_fls], [item[1] for item in avg_all_one]
    avg_rec_stand, avg_rec_mys, avg_rec_fls, avg_rec_all_one = [item[2] for item in avg_all_stand], [item[2] for item in avg_all_mys], [item[2] for item in avg_all_fls], [item[2] for item in avg_all_one]
    return avg_fs_stand, avg_fs_mys, avg_fs_fls, avg_fs_all_one



