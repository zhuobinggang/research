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
    if dic1['targets'][i] == 0 and np.abs(dic1['targets'][i] - dic1['results'][i]) < 0.3 and np.abs(dic2['targets'][i] - dic2['results'][i]) > 0.7:
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

