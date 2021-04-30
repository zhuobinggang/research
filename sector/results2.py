import numpy as np

def analyse(datas, dicname='testdic'):
    precs = []
    recs = []
    f1s = []
    baccs = []
    for data in datas:
        precs.append(data['prec'])
        recs.append(data['rec'])
        f1s.append(data['f1'])
        baccs.append(data['bacc'])
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
    return [
        max(a, b, c) for a, b, c in zip(f123[0::3], f123[1::3], f123[2::3])
    ]


# ============= 辅助函数 ================

standard_cls_2vs2_early = [{
    'prec': 0.7576866764275256,
    'rec': 0.6456643792888335,
    'f1': 0.6972044459413944,
    'bacc': 0.774069667546597,
    'index': 1,
    'valid_loss': 2013.1536898463964
}, {
    'prec': 0.7081545064377682,
    'rec': 0.7205240174672489,
    'f1': 0.7142857142857142,
    'bacc': 0.7901382609434064,
    'index': 1,
    'valid_loss': 1974.137339234352
}, {
    'prec': 0.7400673400673401,
    'rec': 0.6855895196506551,
    'f1': 0.7117875647668396,
    'bacc': 0.7859297038441844,
    'index': 1,
    'valid_loss': 2088.899380683899
}, {
    'prec':
    0.7615499254843517,
    'rec':
    0.6375545851528385,
    'f1':
    0.6940577249575552,
    'bacc':
    0.7716352772552643,
    'index':
    0,
    'valid_loss':
    2020.6016175150871,
    'valid_losses':
    [2020.6016175150871, 2094.7059876322746, 2082.020057745278],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7873376623376623,
    'rec':
    0.6051154086088584,
    'f1':
    0.6843033509700176,
    'bacc':
    0.7639601792602335,
    'index':
    1,
    'valid_loss':
    2022.9384071528912,
    'valid_losses':
    [2990.682630300522, 2022.9384071528912, 2026.8345611840487],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.8434093161546086,
    'rec':
    0.5308796007485964,
    'f1':
    0.6516079632465545,
    'bacc':
    0.742163430309478,
    'index':
    1,
    'valid_loss':
    2073.5598040521145,
    'valid_losses': [2523.514703154564, 2073.5598040521145, 2179.222513630986],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.732706514439221,
    'rec':
    0.6805988771054273,
    'f1':
    0.7056921086675291,
    'bacc':
    0.7816665569970271,
    'index':
    1,
    'valid_loss':
    1989.4620082229376,
    'valid_losses': [2065.730595856905, 1989.4620082229376, 2357.928140513599],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7562544674767692,
    'rec':
    0.6600124766063631,
    'f1':
    0.7048634243837442,
    'bacc':
    0.7797705282265758,
    'index':
    1,
    'valid_loss':
    2001.6503214389086,
    'valid_losses':
    [2062.7921532690525, 2001.6503214389086, 2158.926869429648],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.8140350877192982,
    'rec':
    0.578914535246413,
    'f1':
    0.6766314254465913,
    'bacc':
    0.7582256824729413,
    'index':
    1,
    'valid_loss':
    2051.5572706237435,
    'valid_losses':
    [2107.7913143634796, 2051.5572706237435, 2087.214514002204],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7065152420800956,
    'rec':
    0.7373674360573924,
    'f1':
    0.7216117216117217,
    'bacc':
    0.796350188270299,
    'index':
    1,
    'valid_loss':
    2036.3661597371101,
    'valid_losses':
    [2180.4793668687344, 2036.3661597371101, 2191.5274894982576],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7489481065918654,
    'rec':
    0.6662507797878977,
    'f1':
    0.705183228788379,
    'bacc':
    0.7803852602534067,
    'index':
    0,
    'valid_loss':
    2038.3954220414162,
    'valid_losses': [2038.3954220414162, 2077.291493281722, 2247.473037019372],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7771084337349398,
    'rec':
    0.6437928883343731,
    'f1':
    0.7041965199590584,
    'bacc':
    0.7782900799951182,
    'index':
    1,
    'valid_loss':
    1975.806840479374,
    'valid_losses': [2125.508185148239, 1975.806840479374, 2169.2059966921806],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7291139240506329,
    'rec':
    0.7186525265127885,
    'f1':
    0.7238454288407163,
    'bacc':
    0.7962738177643495,
    'index':
    1,
    'valid_loss':
    1931.3728781342506,
    'valid_losses':
    [2087.621841609478, 1931.3728781342506, 2015.6313183009624],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.763595166163142,
    'rec':
    0.6306924516531504,
    'f1':
    0.6908097027673387,
    'bacc':
    0.7692354420905705,
    'index':
    0,
    'valid_loss':
    2040.4531159102917,
    'valid_losses':
    [2040.4531159102917, 2090.1219539493322, 2291.856302663684],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.6668466522678186,
    'rec':
    0.7704304429195259,
    'f1':
    0.7149059334298119,
    'bacc':
    0.7943195231686611,
    'index':
    0,
    'valid_loss':
    2036.0817962288857,
    'valid_losses': [2036.0817962288857, 2118.413713261485, 2051.535519309342],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7721617418351477,
    'rec':
    0.619463505926388,
    'f1':
    0.6874350986500519,
    'bacc':
    0.7665673451847614,
    'index':
    1,
    'valid_loss':
    2009.0808644741774,
    'valid_losses':
    [2095.7296721339226, 2009.0808644741774, 2323.8955681324005],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.6870359794403198,
    'rec':
    0.7504678727386151,
    'f1':
    0.7173524150268337,
    'bacc':
    0.7945032351318297,
    'index':
    2,
    'valid_loss':
    2156.5195315033197,
    'valid_losses':
    [2355.1862604022026, 2186.691000163555, 2156.5195315033197],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7829209896249002,
    'rec':
    0.6119775421085465,
    'f1':
    0.6869747899159665,
    'bacc':
    0.7659180580312915,
    'index':
    1,
    'valid_loss':
    1980.696169257164,
    'valid_losses': [2193.855884075165, 1980.696169257164, 2149.5138456374407],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.8017543859649123,
    'rec':
    0.5701809107922645,
    'f1':
    0.6664236237695954,
    'bacc':
    0.7517964070755665,
    'index':
    1,
    'valid_loss':
    1986.3672372549772,
    'valid_losses': [2062.817413210869, 1986.3672372549772, 2334.725053332746],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7570422535211268,
    'rec':
    0.670617592014972,
    'f1':
    0.7112140258021833,
    'bacc':
    0.7844838107393657,
    'index':
    1,
    'valid_loss':
    1966.068681448698,
    'valid_losses': [2271.210763156414, 1966.068681448698, 2164.9356861412525],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7623546511627907,
    'rec':
    0.654398003742982,
    'f1':
    0.7042631755622693,
    'bacc':
    0.7790257549651858,
    'index':
    1,
    'valid_loss':
    1978.4621387422085,
    'valid_losses':
    [2116.086154282093, 1978.4621387422085, 2145.4962778389454],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.7003021148036254,
    'rec':
    0.7230193387398628,
    'f1':
    0.7114794352363414,
    'bacc':
    0.7884395456221411,
    'index':
    0,
    'valid_loss':
    2067.5091626644135,
    'valid_losses': [2067.5091626644135, 2149.6319233477116, 2191.17778390646],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}, {
    'prec':
    0.69,
    'rec':
    0.7317529631940112,
    'f1':
    0.7102633969118983,
    'bacc':
    0.7882394751149785,
    'index':
    1,
    'valid_loss':
    2085.7454886734486,
    'valid_losses': [2116.344100818038, 2085.7454886734486, 2319.415672790259],
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop'
}]

sector_2vs2_early_rate02 = [
    {
        'prec': 0.7491090520313614,
        'rec': 0.6556456643792888,
        'f1': 0.699268130405855,
        'bacc': 0.775966615336374,
        'index': 1,
        'valid_loss': 2831.0838955938816
    },
    {
        'prec': 0.764367816091954,
        'rec': 0.6637554585152838,
        'f1': 0.7105175292153589,
        'bacc': 0.7835571635534581,
        'index': 1,
        'valid_loss': 2789.694347858429
    },
    {
        'prec': 0.7475795297372061,
        'rec': 0.6743605739238927,
        'f1': 0.7090849458838964,
        'bacc': 0.783408925736254,
        'index': 1,
        'valid_loss': 2940.132279574871
    },
]
