init_G_Symmetry_Mainichi(half=2, batch=4, mini=False)
m1 = t.load('save/my_0.tch')
m2 = t.load('save/fl_0.tch')
tld = G['testld']
dic1 = {'att': [], 'idss': [], 'results': [], 'targets': [], 'labels': []}
dic2 = {'att': [], 'idss': [], 'results': [], 'targets': [], 'labels': []}
for mass in tld:
    atts, idss, results, targets, labelss = get_att_ours(m1, mass)
    dic1['att'] += atts
    dic1['idss'] += idss
    dic1['results'] += results
    dic1['targets'] += targets
    dic1['labels'] += labelss
    atts, idss, results, targets, labelss = get_att_baseline(m2, mass)
    dic2['att'] += atts
    dic2['idss'] += idss
    dic2['results'] += results
    dic2['targets'] += targets
    dic2['labels'] += labelss
id_value_pairs = []
for i in range(len(dic1['results'])):
    # if dic1['targets'][i] == 0 and np.abs(dic1['targets'][i] - dic1['results'][i]) < 0.3 and np.abs(dic2['targets'][i] - dic2['results'][i]) > 0.7:
    if np.abs(dic1['targets'][i] - dic1['results'][i]) < 0.3 and np.abs(dic2['targets'][i] - dic2['results'][i]) > 0.7:
        id_value_pairs.append((i, np.abs(dic1['results'][i] - dic2['results'][i])))
    # return id_value_pairs, dic1, dic2
    # SHOW
idss = dic1['idss'][388]
att1 = dic1['att'][388]
att2 = dic2['att'][388]
for idx, a1, a2 in zip(idss, att1, att2):
    print(f'{m1.toker.decode(idx)}: {round(a1, 3)}, {round(a2, 3)}')


id_value_pairs = []
count = 0
ids = []
for i in range(len(dic1['results'])):
    labels = dic1['labels'][i]
    if len(labels) == 4 and labels[2] == 1: 
        count += 1


outputs = []
for i in range(len(dic1['results'])):
    labels = dic1['labels'][i]
    if len(labels) == 4 and labels[2] == 1 and labels[1] == 0: 
        outputs.append(dic1['results'][i])


outputs2 = []
for i in range(len(dic2['results'])):
    labels = dic2['labels'][i]
    if len(labels) == 4 and labels[2] == 1 and labels[1] == 0: 
        outputs2.append(dic2['results'][i])


[(57, 0.438544899225235), (718, 0.4401474595069885), (1610, 0.4860328435897827), (1057, 0.4886564612388611), (5891, 0.4892771393060684), (2284, 0.49094358086586), (5410, 0.5023594200611115), (5430, 0.509790450334549), (3486, 0.5328832566738129), (5527, 0.533017486333847), (2385, 0.5377205610275269), (6696, 0.5698126256465912), (6463, 0.5713282376527786), (1214, 0.5872098952531815), (886, 0.6055899560451508), (4190, 0.6114858090877533), (5156, 0.6210010647773743), (4478, 0.6240078881382942), (156, 0.6273315846920013), (1587, 0.6309479773044586), (2177, 0.6394941508769989), (1858, 0.6410343274474144), (3048, 0.6452006176114082), (5679, 0.6557422392070293), (3783, 0.6669213026762009), (3196, 0.6702942624688148), (2163, 0.6741553135216236), (4503, 0.6829911917448044), (6132, 0.6859948709607124), (1733, 0.6893036067485809), (1363, 0.69074104167521), (1951, 0.7033011596649885), (4600, 0.7147884853184223), (3809, 0.7222359478473663), (820, 0.7284649796783924), (3868, 0.7344106882810593), (1366, 0.7527086790651083)]


# 平均重要token占比
pers1 = []
pers2 = []
for i in range(len(dic1['results'])):
    atts1 = dic1['att'][i]
    if len(atts1) == 0:
        pass
    else:
        atts2 = dic2['att'][i]
        count1 = sum([1 for att in atts1 if att > (1 / len(atts1))])
        count2 = sum([1 for att in atts2 if att > (1 / len(atts2))])
        per1 = count1 / len(atts1)
        pers1.append(per1)
        per2 = count2 / len(atts2)
        pers2.append(per2)


## 3v3

for i in range(5):
    idx = i
    m = t.load(f'save/my_3v3_{idx}.tch')
    outputs, targets = get_test_result(m, tld)
    g_save['targets'] = targets
    g_save['outputs'].append(outputs)


avg_fs_stand = []
for sampled_indexs in sampled_multi_indexs:
    avg_fs.append(cal_avg_f_score(targets, g_save['outputs'], sampled_indexs))



avg_prec_stand, avg_prec_mys, avg_prec_fls, avg_prec_all_one = [item[1] for item in avg_all_stand], [item[1] for item in avg_all_mys], [item[1] for item in avg_all_fls], [item[1] for item in avg_all_one]

avg_rec_stand, avg_rec_mys, avg_rec_fls, avg_rec_all_one = [item[2] for item in avg_all_stand], [item[2] for item in avg_all_mys], [item[2] for item in avg_all_fls], [item[2] for item in avg_all_one]
