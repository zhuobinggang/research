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


def print_out(r):
    e0 = []
    e1 = []
    e2 = []
    for item in r:
        e0.append(item[0]['f1'])
        e1.append(item[1]['f1'])
        e2.append(item[2]['f1'])
    print(f'{round(np.average(e0), 5)}, {round(np.average(e1), 5)}, {round(np.average(e2), 5)}')

