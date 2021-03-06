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


def get_fs_by_mess(raw, step=2):
    results = []
    _, _, fs, _ = analyse(raw)
    for i in range(step):
        results.append(fs[i::step])
    return results


def get_index_count(raw, len_max=20, return_dic=True):
    indexs = [item['index'] for item in raw]
    indexs = indexs[:20]
    dic = {
        '0': len([idx for idx in indexs if idx == 0]),
        '1': len([idx for idx in indexs if idx == 1]),
        '2': len([idx for idx in indexs if idx == 2])
    }
    if return_dic:
        return dic
    else:
        return f"{dic['0']} {dic['1']} {dic['2']}"


def split_mess(raw, step):
    results = []
    for i in range(step):
        results.append(raw[i::step])
    return results


def get_all_mess():
    std = standard_cls_2vs2_early
    r01, r03, r04 = split_mess(rate_mess2, 3)
    r02, r08, r10, r15 = split_mess(rate_test_mess, 4)
    r05 = sector_2vs2_early_rate05
    return [std, r01, r02, r03, r04, r05, r08, r10, r15]


def draw_converge_chart():
    results = get_all_mess()
    idxs = [get_index_count(raw, 20, False) for raw in results]
    return idxs


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

sector_2vs2_early_rate05 = [{
    'prec':
    0.7333333333333333,
    'rec':
    0.6999376169681847,
    'f1':
    0.7162464091924673,
    'bacc':
    0.7898627389496198,
    'index':
    1,
    'valid_loss':
    4327.87307703495,
    'valid_losses': [4345.722040891647, 4327.87307703495, 5519.813937470317],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7835463258785943,
    'rec':
    0.6119775421085465,
    'f1':
    0.687215411558669,
    'bacc':
    0.7660653768291701,
    'index':
    0,
    'valid_loss':
    4139.549301505089,
    'valid_losses': [4139.549301505089, 4564.369498342276, 5735.163819000125],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7725426482534524,
    'rec':
    0.5932626325639426,
    'f1':
    0.671136203246295,
    'bacc':
    0.7553820528759607,
    'index':
    0,
    'valid_loss':
    4176.3977382183075,
    'valid_losses': [4176.3977382183075, 4463.09445258975, 5482.3004958331585],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7420907840440165,
    'rec':
    0.6731129132875858,
    'f1':
    0.7059208374223096,
    'bacc':
    0.7813119074393144,
    'index':
    0,
    'valid_loss':
    4064.9906170368195,
    'valid_losses': [4064.9906170368195, 5037.436606556177, 5697.142442628741],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.6902963393375944,
    'rec':
    0.7411104179663132,
    'f1':
    0.7148014440433214,
    'bacc':
    0.7920342897138578,
    'index':
    1,
    'valid_loss':
    4282.6700212955475,
    'valid_losses': [4546.273358464241, 4282.6700212955475, 4867.585405558348],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7584471603163192,
    'rec':
    0.6581409856519027,
    'f1':
    0.7047428189712759,
    'bacc':
    0.7795713767387387,
    'index':
    1,
    'valid_loss':
    4288.63455080986,
    'valid_losses': [4408.9548716545105, 4288.63455080986, 5038.31439127028],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7510259917920656,
    'rec':
    0.6849656893325016,
    'f1':
    0.7164763458401305,
    'bacc':
    0.7888588022384371,
    'index':
    0,
    'valid_loss':
    4075.737256228924,
    'valid_losses': [4075.737256228924, 4447.060751795769, 5470.298394605517],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7228682170542635,
    'rec':
    0.6980661260137243,
    'f1':
    0.710250714059029,
    'bacc':
    0.7858332987169387,
    'index':
    1,
    'valid_loss':
    4259.321000397205,
    'valid_losses': [4395.793573498726, 4259.321000397205, 5390.6001929193735],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7505668934240363,
    'rec':
    0.619463505926388,
    'f1':
    0.6787423103212578,
    'bacc':
    0.7611165496632529,
    'index':
    0,
    'valid_loss':
    4176.120030045509,
    'valid_losses': [4176.120030045509, 4490.892331123352, 5181.815538018942],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7323446327683616,
    'rec':
    0.6469120399251403,
    'f1':
    0.6869824445180523,
    'bacc':
    0.7676221955665772,
    'index':
    0,
    'valid_loss':
    4386.949203252792,
    'valid_losses': [4386.949203252792, 4461.467592984438, 5459.6935785114765],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7944630872483222,
    'rec':
    0.5907673112913288,
    'f1':
    0.6776386404293381,
    'bacc':
    0.759290550165405,
    'index':
    1,
    'valid_loss':
    4157.189886271954,
    'valid_losses': [6062.060273647308, 4157.189886271954, 4992.48538698256],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7798092209856916,
    'rec':
    0.6119775421085465,
    'f1':
    0.6857742048234883,
    'bacc':
    0.7651814640418985,
    'index':
    0,
    'valid_loss':
    4357.340858340263,
    'valid_losses': [4357.340858340263, 4396.17576020956, 5298.7876684218645],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.6780346820809249,
    'rec':
    0.7317529631940112,
    'f1':
    0.7038703870387039,
    'bacc':
    0.7838199111786202,
    'index':
    0,
    'valid_loss':
    4254.285311698914,
    'valid_losses': [4254.285311698914, 4453.267217963934, 5296.663795799017],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7330623306233063,
    'rec':
    0.6749844042420462,
    'f1':
    0.7028255927249107,
    'bacc':
    0.7794485957568511,
    'index':
    1,
    'valid_loss':
    4281.8996178507805,
    'valid_losses': [5490.452702999115, 4281.8996178507805, 4845.096800267696],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7458448753462604,
    'rec':
    0.6718652526512788,
    'f1':
    0.7069248441089596,
    'bacc':
    0.7818666275041898,
    'index':
    0,
    'valid_loss':
    4166.445082604885,
    'valid_losses': [4166.445082604885, 5187.185514479876, 5811.650923274457],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.6954756380510441,
    'rec':
    0.7479725514660013,
    'f1':
    0.7207694619777578,
    'bacc':
    0.7966439068467308,
    'index':
    0,
    'valid_loss':
    4141.961904466152,
    'valid_losses': [4141.961904466152, 4910.5625423491, 5627.069515764713],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7111251580278128,
    'rec':
    0.7018091079226451,
    'f1':
    0.706436420722135,
    'bacc':
    0.7835798633307981,
    'index':
    0,
    'valid_loss':
    4257.183497071266,
    'valid_losses': [4257.183497071266, 4439.644009023905, 5577.119589909911],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.738795518207283,
    'rec':
    0.6581409856519027,
    'f1':
    0.6961398878258002,
    'bacc':
    0.77412058121723,
    'index':
    0,
    'valid_loss':
    4320.480976045132,
    'valid_losses': [4320.480976045132, 4340.609968036413, 5209.5207595825195],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.6984318455971049,
    'rec':
    0.7223955084217093,
    'f1':
    0.7102115915363384,
    'bacc':
    0.78753835527155,
    'index':
    0,
    'valid_loss':
    4264.165305733681,
    'valid_losses': [4264.165305733681, 4462.368834346533, 5489.5281620025635],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}, {
    'prec':
    0.7595204513399154,
    'rec':
    0.6718652526512788,
    'f1':
    0.7130089374379345,
    'bacc':
    0.7856969162490337,
    'index':
    1,
    'valid_loss':
    4178.1119602918625,
    'valid_losses': [4552.590218901634, 4178.1119602918625, 4994.409468099475],
    'desc':
    'Sector_Split 2vs2 rate=0.5 early_stop'
}]

rate_test_mess = [{
    'prec': 0.7657657657657657,
    'rec': 0.6363069245165315,
    'f1': 0.6950596252129473,
    'bacc': 0.7721899973201396,
    'index': 0,
    'valid_loss': 2955.2725541591644,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7269736842105263,
    'rec': 0.6893325015595758,
    'f1': 0.7076528978546271,
    'bacc': 0.7835289496601651,
    'index': 1,
    'valid_loss': 5784.560802876949,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7963125548726954,
    'rec': 0.5658140985651903,
    'f1': 0.6615609044493072,
    'bacc': 0.7487290881747577,
    'index': 1,
    'valid_loss': 6626.2916342020035,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7093167701863354,
    'rec': 0.7124142233312539,
    'f1': 0.7108621226268287,
    'bacc': 0.7872619142584378,
    'index': 0,
    'valid_loss': 8351.71722984314,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7452445652173914,
    'rec': 0.6843418590143481,
    'f1': 0.7134959349593495,
    'bacc': 0.7869263803026956,
    'index': 0,
    'valid_loss': 2851.70724517107,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.758854559155991,
    'rec': 0.6281971303805365,
    'f1': 0.6873720136518772,
    'bacc': 0.7669565498691133,
    'index': 0,
    'valid_loss': 5542.039062976837,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.6904326630103595,
    'rec': 0.7067997504678727,
    'f1': 0.6985203452527743,
    'bacc': 0.7785619259116028,
    'index': 0,
    'valid_loss': 6464.697604894638,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7909238249594813,
    'rec': 0.6088583905177791,
    'f1': 0.6880507578427916,
    'bacc': 0.7664209454062083,
    'index': 0,
    'valid_loss': 8707.408022403717,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7030231179608772,
    'rec': 0.7398627573300063,
    'f1': 0.7209726443768997,
    'bacc': 0.7961246609278199,
    'index': 1,
    'valid_loss': 2884.5617750883102,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7293964957819598,
    'rec': 0.7011852776044916,
    'f1': 0.7150127226463103,
    'bacc': 0.7891607000868657,
    'index': 1,
    'valid_loss': 5595.761509776115,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7292889758643183,
    'rec': 0.6974422956955708,
    'f1': 0.7130102040816327,
    'bacc': 0.7875838467281626,
    'index': 0,
    'valid_loss': 6563.623757004738,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7454923717059639,
    'rec': 0.670617592014972,
    'f1': 0.7060755336617406,
    'bacc': 0.7812427971860363,
    'index': 1,
    'valid_loss': 8867.819945633411,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7058823529411765,
    'rec': 0.72613849033063,
    'f1': 0.7158671586715867,
    'bacc': 0.7916196281941895,
    'index': 1,
    'valid_loss': 2880.572885453701,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7607318789584799,
    'rec': 0.6743605739238927,
    'f1': 0.71494708994709,
    'bacc': 0.7870918956832191,
    'index': 1,
    'valid_loss': 5680.746868252754,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7342430149447693,
    'rec': 0.7049282595134123,
    'f1': 0.7192870782940801,
    'bacc': 0.7922107414243549,
    'index': 0,
    'valid_loss': 6421.55176627636,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7175618587809294,
    'rec': 0.7417342482844667,
    'f1': 0.7294478527607362,
    'bacc': 0.8019219267350441,
    'index': 0,
    'valid_loss': 8472.758272051811,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.6916325336454067,
    'rec': 0.7373674360573924,
    'f1': 0.713768115942029,
    'bacc': 0.7910467115466691,
    'index': 0,
    'valid_loss': 2782.777784496546,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7413441955193483,
    'rec': 0.6812227074235808,
    'f1': 0.7100130039011704,
    'bacc': 0.7844828917200403,
    'index': 1,
    'valid_loss': 5890.698537707329,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7079207920792079,
    'rec': 0.7136618839675608,
    'f1': 0.7107797452625039,
    'bacc': 0.7872964693850768,
    'index': 0,
    'valid_loss': 6361.033912181854,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7910562837316885,
    'rec': 0.6400499064254522,
    'f1': 0.7075862068965516,
    'bacc': 0.780101558987623,
    'index': 0,
    'valid_loss': 8530.864558458328,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7371202113606341,
    'rec': 0.6961946350592639,
    'f1': 0.716073147256978,
    'bacc': 0.7894644359739454,
    'index': 1,
    'valid_loss': 2890.9718214273453,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7717872968980798,
    'rec': 0.6519026824703681,
    'f1': 0.706797429827528,
    'bacc': 0.7804298326906938,
    'index': 1,
    'valid_loss': 5586.585286796093,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7478932584269663,
    'rec': 0.6643792888334373,
    'f1': 0.7036669970267592,
    'bacc': 0.779302195978298,
    'index': 0,
    'valid_loss': 6426.383313298225,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.6927239927841251,
    'rec': 0.7186525265127885,
    'f1': 0.7054500918554807,
    'bacc': 0.7840463575404248,
    'index': 0,
    'valid_loss': 8823.928306102753,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7340136054421769,
    'rec': 0.6731129132875858,
    'f1': 0.7022453628376181,
    'bacc': 0.7789548066732567,
    'index': 1,
    'valid_loss': 2895.904962748289,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7681632653061224,
    'rec': 0.587024329382408,
    'f1': 0.6654879773691655,
    'bacc': 0.7516736260936789,
    'index': 0,
    'valid_loss': 5514.849280834198,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7232202262142382,
    'rec': 0.6781035558328135,
    'f1': 0.699935608499678,
    'bacc': 0.7777671579989053,
    'index': 0,
    'valid_loss': 6163.074236750603,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7379790940766551,
    'rec': 0.6606363069245166,
    'f1': 0.6971691902567478,
    'bacc': 0.7749262854599012,
    'index': 1,
    'valid_loss': 8617.478661060333,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7017441860465117,
    'rec': 0.7529631940112289,
    'f1': 0.7264520012037315,
    'bacc': 0.8009070536938878,
    'index': 1,
    'valid_loss': 2949.1655084490776,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7269021739130435,
    'rec': 0.6674984404242046,
    'f1': 0.6959349593495935,
    'bacc': 0.7745270634649013,
    'index': 0,
    'valid_loss': 5628.853838920593,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7284390591589451,
    'rec': 0.6375545851528385,
    'f1': 0.679973386560213,
    'bacc': 0.762648830584669,
    'index': 0,
    'valid_loss': 6563.302649974823,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.6888111888111889,
    'rec': 0.7373674360573924,
    'f1': 0.7122627297378729,
    'bacc': 0.7900154799615189,
    'index': 1,
    'valid_loss': 8706.936002492905,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7354497354497355,
    'rec': 0.6936993137866501,
    'f1': 0.7139646869983949,
    'bacc': 0.7879221377418812,
    'index': 1,
    'valid_loss': 2876.9219871759415,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.810692375109553,
    'rec': 0.5770430442919526,
    'f1': 0.674198250728863,
    'bacc': 0.7567006618041967,
    'index': 0,
    'valid_loss': 5604.580413579941,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7230077120822622,
    'rec': 0.7018091079226451,
    'f1': 0.7122507122507121,
    'bacc': 0.7874101520756419,
    'index': 0,
    'valid_loss': 6140.461759328842,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7623688155922039,
    'rec': 0.6344354335620711,
    'f1': 0.6925434116445354,
    'bacc': 0.7705176578535164,
    'index': 0,
    'valid_loss': 8469.23231112957,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7726597325408618,
    'rec': 0.6487835308796007,
    'f1': 0.7053238385893523,
    'bacc': 0.7793122132889458,
    'index': 1,
    'valid_loss': 2859.4761095643044,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7874080130825838,
    'rec': 0.6007485963817841,
    'f1': 0.6815286624203822,
    'bacc': 0.7620714107424537,
    'index': 0,
    'valid_loss': 5570.202001810074,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7552123552123552,
    'rec': 0.6101060511540861,
    'f1': 0.6749482401656315,
    'bacc': 0.7583529666495239,
    'index': 0,
    'valid_loss': 6456.287269473076,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7343857240905971,
    'rec': 0.6674984404242046,
    'f1': 0.6993464052287581,
    'bacc': 0.7767368454330805,
    'index': 0,
    'valid_loss': 8580.131745934486,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7463576158940397,
    'rec': 0.7030567685589519,
    'f1': 0.7240603919049149,
    'bacc': 0.7951052846919686,
    'index': 1,
    'valid_loss': 2853.2058585882187,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7354352296093215,
    'rec': 0.669369931378665,
    'f1': 0.7008491182233834,
    'bacc': 0.7778199097081893,
    'index': 0,
    'valid_loss': 5398.284265041351,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7461322081575246,
    'rec': 0.6618839675608235,
    'f1': 0.7014876033057851,
    'bacc': 0.7777598977462338,
    'index': 1,
    'valid_loss': 6478.163499355316,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7549019607843137,
    'rec': 0.6724890829694323,
    'f1': 0.7113163972286374,
    'bacc': 0.7846829622272029,
    'index': 1,
    'valid_loss': 8465.23497235775,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7260362694300518,
    'rec': 0.6993137866500312,
    'f1': 0.7124245312996504,
    'bacc': 0.7873410418223639,
    'index': 1,
    'valid_loss': 2884.631474494934,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7559607293127629,
    'rec': 0.6724890829694323,
    'f1': 0.7117860680092438,
    'bacc': 0.7849775998229601,
    'index': 1,
    'valid_loss': 5532.020818591118,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7285714285714285,
    'rec': 0.6681222707423581,
    'f1': 0.6970387243735763,
    'bacc': 0.775280935017614,
    'index': 0,
    'valid_loss': 6262.573616743088,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7480809490579204,
    'rec': 0.6687461010605116,
    'f1': 0.7061923583662714,
    'bacc': 0.7811909644960778,
    'index': 0,
    'valid_loss': 8626.839842438698,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7771982116244411,
    'rec': 0.6506550218340611,
    'f1': 0.7083191850594228,
    'bacc': 0.7812791903513263,
    'index': 1,
    'valid_loss': 2894.175076186657,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7414473684210526,
    'rec': 0.7030567685589519,
    'f1': 0.7217419148254883,
    'bacc': 0.7936320967131825,
    'index': 1,
    'valid_loss': 5593.390115916729,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7508674531575295,
    'rec': 0.6749844042420462,
    'f1': 0.7109067017082786,
    'bacc': 0.7846047536826024,
    'index': 1,
    'valid_loss': 6406.103041887283,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.6911764705882353,
    'rec': 0.7330006238303182,
    'f1': 0.7114744171964881,
    'bacc': 0.7891579430288892,
    'index': 0,
    'valid_loss': 8473.823615312576,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7988215488215489,
    'rec': 0.5920149719276356,
    'f1': 0.6800429953421713,
    'bacc': 0.7607982932708302,
    'index': 1,
    'valid_loss': 2898.4569732546806,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7501875468867217,
    'rec': 0.6238303181534622,
    'f1': 0.6811989100817438,
    'bacc': 0.7628579993831542,
    'index': 0,
    'valid_loss': 5650.11295735836,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.6801120448179272,
    'rec': 0.7573300062383032,
    'f1': 0.7166469893742622,
    'bacc': 0.7945459695304656,
    'index': 0,
    'valid_loss': 6367.3870795965195,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7188478396994364,
    'rec': 0.7161572052401747,
    'f1': 0.7175,
    'bacc': 0.7919324623725917,
    'index': 1,
    'valid_loss': 8357.223249316216,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7157960199004975,
    'rec': 0.718028696194635,
    'f1': 0.7169106197446278,
    'bacc': 0.7916896574667931,
    'index': 1,
    'valid_loss': 2874.6691220402718,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.6881844380403458,
    'rec': 0.7448533998752339,
    'f1': 0.7153984421809467,
    'bacc': 0.7927272302852892,
    'index': 1,
    'valid_loss': 5586.344415545464,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7211231652839821,
    'rec': 0.7049282595134123,
    'f1': 0.7129337539432177,
    'bacc': 0.788085815083754,
    'index': 1,
    'valid_loss': 6622.682622313499,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7221502590673575,
    'rec': 0.6955708047411104,
    'f1': 0.7086113759135685,
    'bacc': 0.7845856380806318,
    'index': 1,
    'valid_loss': 8587.158179163933,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7306701030927835,
    'rec': 0.7074235807860262,
    'f1': 0.7188589540412044,
    'bacc': 0.7921325328797544,
    'index': 0,
    'valid_loss': 2892.6356108784676,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7038805970149253,
    'rec': 0.735495945102932,
    'f1': 0.7193410616229408,
    'bacc': 0.7946778488036759,
    'index': 1,
    'valid_loss': 5997.4437482357025,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7420470262793915,
    'rec': 0.669369931378665,
    'f1': 0.7038373237126927,
    'bacc': 0.7797350540806112,
    'index': 1,
    'valid_loss': 6699.247068166733,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7783333333333333,
    'rec': 0.5826575171553338,
    'f1': 0.6664288262575812,
    'bacc': 0.7521419583419569,
    'index': 0,
    'valid_loss': 8506.354994535446,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.6966551326412919,
    'rec': 0.7535870243293824,
    'f1': 0.7240035960443512,
    'bacc': 0.7993038244805427,
    'index': 1,
    'valid_loss': 2889.6164675056934,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7002915451895044,
    'rec': 0.7492202121023082,
    'f1': 0.7239300783604582,
    'bacc': 0.7988882439415489,
    'index': 0,
    'valid_loss': 5423.117594957352,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7182284980744544,
    'rec': 0.6980661260137243,
    'f1': 0.7080037962670043,
    'bacc': 0.7843601107381526,
    'index': 1,
    'valid_loss': 6374.982995271683,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7328348062542488,
    'rec': 0.6724890829694323,
    'f1': 0.701366297983084,
    'bacc': 0.7783482539184228,
    'index': 1,
    'valid_loss': 8917.62358391285,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7248576850094877,
    'rec': 0.7149095446038678,
    'f1': 0.7198492462311558,
    'bacc': 0.7933710952247388,
    'index': 1,
    'valid_loss': 2874.305331528187,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7381596752368065,
    'rec': 0.6805988771054273,
    'f1': 0.7082116196040247,
    'bacc': 0.7832870637736918,
    'index': 1,
    'valid_loss': 5439.491467058659,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7437794216543376,
    'rec': 0.6899563318777293,
    'f1': 0.7158576051779936,
    'bacc': 0.7888497039471145,
    'index': 0,
    'valid_loss': 6375.531591296196,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7676848874598071,
    'rec': 0.5957579538365565,
    'f1': 0.6708816297857393,
    'bacc': 0.7553038443313602,
    'index': 0,
    'valid_loss': 8737.119146108627,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.7674247982391783,
    'rec': 0.6525265127885215,
    'f1': 0.7053270397842211,
    'bacc': 0.7795631974667416,
    'index': 1,
    'valid_loss': 2824.3151036798954,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.7200759974667511,
    'rec': 0.7092950717404866,
    'f1': 0.7146448774355751,
    'bacc': 0.789532627207898,
    'index': 0,
    'valid_loss': 5353.5099619627,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7371202113606341,
    'rec': 0.6961946350592639,
    'f1': 0.716073147256978,
    'bacc': 0.7894644359739454,
    'index': 1,
    'valid_loss': 6588.311538755894,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7775919732441472,
    'rec': 0.5801621958827199,
    'f1': 0.6645230439442659,
    'bacc': 0.7508942977056499,
    'index': 0,
    'valid_loss': 8597.053354859352,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}, {
    'prec': 0.761757902852737,
    'rec': 0.6163443543356207,
    'f1': 0.6813793103448275,
    'bacc': 0.7626506686233201,
    'index': 1,
    'valid_loss': 2791.979343265295,
    'desc': 'Sector_Split 2vs2 rate=0.2 early_stop'
}, {
    'prec': 0.75,
    'rec': 0.67747972551466,
    'f1': 0.7118977384464111,
    'bacc': 0.7854104579252734,
    'index': 0,
    'valid_loss': 5453.745950996876,
    'desc': 'Sector_Split 2vs2 rate=0.8 early_stop'
}, {
    'prec': 0.7680945347119645,
    'rec': 0.6487835308796007,
    'f1': 0.7034156239431857,
    'bacc': 0.778133662905917,
    'index': 1,
    'valid_loss': 6375.271540284157,
    'desc': 'Sector_Split 2vs2 rate=1.0 early_stop'
}, {
    'prec': 0.7496570644718793,
    'rec': 0.6818465377417342,
    'f1': 0.7141457040182947,
    'bacc': 0.7871519076451747,
    'index': 0,
    'valid_loss': 8427.014657378197,
    'desc': 'Sector_Split 2vs2 rate=1.5 early_stop'
}]

rate_mess2 = [
    {
        'prec': 0.7755847953216374,
        'rec': 0.6618839675608235,
        'f1': 0.7142376304274655,
        'bacc': 0.7857151128316786,
        'index': 1,
        'valid_loss': 2392.080699414015,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7531734837799718,
        'rec': 0.6662507797878977,
        'f1': 0.7070506454816287,
        'bacc': 0.7815638106364355,
        'index': 1,
        'valid_loss': 3266.6353881061077,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.704225352112676,
        'rec': 0.7174048658764816,
        'f1': 0.7107540173053151,
        'bacc': 0.7875474535628725,
        'index': 0,
        'valid_loss': 3812.9410130381584,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7812257564003103,
        'rec': 0.6281971303805365,
        'f1': 0.6964038727524205,
        'bacc': 0.7725546641885004,
        'index': 2,
        'valid_loss': 2403.3102717399597,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7803223070398643,
        'rec': 0.5739238927011853,
        'f1': 0.6613946800862689,
        'bacc': 0.7488063777000329,
        'index': 0,
        'valid_loss': 3428.100771486759,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.6924882629107981,
        'rec': 0.7361197754210854,
        'f1': 0.7136377381312368,
        'bacc': 0.7908648376221514,
        'index': 0,
        'valid_loss': 3813.9203919172287,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7617354196301565,
        'rec': 0.6681222707423581,
        'f1': 0.7118644067796611,
        'bacc': 0.7847093380818448,
        'index': 1,
        'valid_loss': 2424.2161213457584,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7386138613861386,
        'rec': 0.6980661260137243,
        'f1': 0.7177677998717126,
        'bacc': 0.7906948190469328,
        'index': 1,
        'valid_loss': 3318.729059368372,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7788235294117647,
        'rec': 0.619463505926388,
        'f1': 0.6900625434329395,
        'bacc': 0.7681878519614262,
        'index': 1,
        'valid_loss': 3850.3895722031593,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7368082368082368,
        'rec': 0.7142857142857143,
        'f1': 0.725372188786823,
        'bacc': 0.7968894688105059,
        'index': 1,
        'valid_loss': 2404.545118778944,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7125147579693034,
        'rec': 0.7529631940112289,
        'f1': 0.732180770397331,
        'bacc': 0.8047373424387317,
        'index': 1,
        'valid_loss': 3352.1015244722366,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.6908990011098779,
        'rec': 0.7766687461010605,
        'f1': 0.7312775330396476,
        'bacc': 0.8062778026321449,
        'index': 0,
        'valid_loss': 3941.1303586363792,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7083839611178615,
        'rec': 0.727386150966937,
        'f1': 0.7177593105570945,
        'bacc': 0.7929800525017361,
        'index': 1,
        'valid_loss': 2465.4229112267494,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.769170579029734,
        'rec': 0.6132252027448534,
        'f1': 0.6824019437695245,
        'bacc': 0.763153555998237,
        'index': 0,
        'valid_loss': 3362.4916926920414,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7016706443914081,
        'rec': 0.7336244541484717,
        'f1': 0.717291857273559,
        'bacc': 0.7931528281349312,
        'index': 1,
        'valid_loss': 3898.020197868347,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7017650639074863,
        'rec': 0.7192763568309419,
        'f1': 0.7104128157732592,
        'bacc': 0.7874519674549524,
        'index': 1,
        'valid_loss': 2469.1206278800964,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.6818181818181818,
        'rec': 0.7205240174672489,
        'f1': 0.7006369426751592,
        'bacc': 0.780857176677054,
        'index': 0,
        'valid_loss': 3347.0688635110855,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7163995067817509,
        'rec': 0.7248908296943232,
        'f1': 0.7206201550387598,
        'bacc': 0.7946787678230013,
        'index': 1,
        'valid_loss': 3910.7902151942253,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.6936781609195403,
        'rec': 0.7529631940112289,
        'f1': 0.7221058929105594,
        'bacc': 0.7979606777363157,
        'index': 1,
        'valid_loss': 2467.27640286088,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7412491420727523,
        'rec': 0.6737367436057392,
        'f1': 0.7058823529411764,
        'bacc': 0.7813291850026338,
        'index': 1,
        'valid_loss': 3485.9509076178074,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7525286581254215,
        'rec': 0.6961946350592639,
        'f1': 0.7232663642255348,
        'bacc': 0.7940313187081823,
        'index': 1,
        'valid_loss': 3720.3732565045357,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7278521351179095,
        'rec': 0.7124142233312539,
        'f1': 0.7200504413619166,
        'bacc': 0.7933019849714608,
        'index': 0,
        'valid_loss': 2382.0337403416634,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7132352941176471,
        'rec': 0.72613849033063,
        'f1': 0.7196290571870169,
        'bacc': 0.7941240477581258,
        'index': 1,
        'valid_loss': 3365.4881826639175,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.8019891500904159,
        'rec': 0.553337492202121,
        'f1': 0.6548541897379107,
        'bacc': 0.7444059293656451,
        'index': 0,
        'valid_loss': 3942.3375267982483,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7316421895861148,
        'rec': 0.6837180286961946,
        'f1': 0.7068687520154788,
        'bacc': 0.7826368576008964,
        'index': 1,
        'valid_loss': 2488.1529910564423,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.770540340488527,
        'rec': 0.6494073611977542,
        'f1': 0.7048070412999323,
        'bacc': 0.7790348532565082,
        'index': 1,
        'valid_loss': 3356.2160817086697,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7452631578947368,
        'rec': 0.662507797878977,
        'f1': 0.7014531043593132,
        'bacc': 0.7777771753095533,
        'index': 0,
        'valid_loss': 3797.9383860826492,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7184343434343434,
        'rec': 0.7099189020586401,
        'f1': 0.7141512394101035,
        'bacc': 0.7892552671754602,
        'index': 1,
        'valid_loss': 2427.90736143291,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7527397260273972,
        'rec': 0.6855895196506551,
        'f1': 0.7175971269996736,
        'bacc': 0.7896126737911495,
        'index': 2,
        'valid_loss': 3435.8015691041946,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7633642195295794,
        'rec': 0.6681222707423581,
        'f1': 0.7125748502994012,
        'bacc': 0.7851512944754808,
        'index': 1,
        'valid_loss': 3798.3826847672462,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7322033898305085,
        'rec': 0.6737367436057392,
        'f1': 0.7017543859649124,
        'bacc': 0.778677446640819,
        'index': 0,
        'valid_loss': 2495.2323710918427,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7492857142857143,
        'rec': 0.654398003742982,
        'f1': 0.6986346986346987,
        'bacc': 0.7754901038160991,
        'index': 0,
        'valid_loss': 3369.140181183815,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7930476960388035,
        'rec': 0.6119775421085465,
        'f1': 0.6908450704225352,
        'bacc': 0.7682751587973493,
        'index': 0,
        'valid_loss': 3767.104766368866,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7573371510379384,
        'rec': 0.6600124766063631,
        'f1': 0.7053333333333333,
        'bacc': 0.780065165822333,
        'index': 1,
        'valid_loss': 2415.2152996361256,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7565485362095532,
        'rec': 0.6126013724267,
        'f1': 0.6770079283005861,
        'bacc': 0.7597479460837094,
        'index': 0,
        'valid_loss': 3366.0176050662994,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7195967233774417,
        'rec': 0.7124142233312539,
        'f1': 0.715987460815047,
        'bacc': 0.7906502466096459,
        'index': 1,
        'valid_loss': 3903.1329305171967,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.631155303030303,
        'rec': 0.8315658140985652,
        'f1': 0.7176312247644685,
        'bacc': 0.801021563501846,
        'index': 0,
        'valid_loss': 2560.4776350855827,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.6454081632653061,
        'rec': 0.7891453524641298,
        'f1': 0.710075778838058,
        'bacc': 0.7921861117064314,
        'index': 0,
        'valid_loss': 3465.645968556404,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7541100786275912,
        'rec': 0.6581409856519027,
        'f1': 0.7028647568287809,
        'bacc': 0.7783928263557098,
        'index': 1,
        'valid_loss': 3848.4511507749557,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.8055299539170507,
        'rec': 0.545227698066126,
        'f1': 0.6502976190476191,
        'bacc': 0.7415295826806765,
        'index': 1,
        'valid_loss': 2575.615545064211,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.75,
        'rec': 0.6250779787897692,
        'f1': 0.6818645797890438,
        'bacc': 0.7633345109034291,
        'index': 0,
        'valid_loss': 3401.0479414463043,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7550287356321839,
        'rec': 0.6556456643792888,
        'f1': 0.7018363939899834,
        'bacc': 0.7775871221130386,
        'index': 0,
        'valid_loss': 3900.5296812057495,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7186311787072244,
        'rec': 0.7074235807860262,
        'f1': 0.7129833385727757,
        'bacc': 0.7883022441349106,
        'index': 2,
        'valid_loss': 2658.714612454176,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7597305389221557,
        'rec': 0.6331877729257642,
        'f1': 0.6907111262334128,
        'bacc': 0.7693045523438485,
        'index': 0,
        'valid_loss': 3313.9476650357246,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7841105354058722,
        'rec': 0.5664379288833438,
        'f1': 0.6577327055414706,
        'bacc': 0.7463892649720196,
        'index': 0,
        'valid_loss': 3722.73853379488,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7468443197755961,
        'rec': 0.6643792888334373,
        'f1': 0.7032023770221194,
        'bacc': 0.7790075583825407,
        'index': 1,
        'valid_loss': 2398.0701675713062,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.6950565812983919,
        'rec': 0.7280099812850904,
        'f1': 0.7111517367458865,
        'bacc': 0.7885777661286972,
        'index': 0,
        'valid_loss': 3377.7753264307976,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.6866781017724414,
        'rec': 0.7492202121023082,
        'f1': 0.7165871121718377,
        'bacc': 0.7938794048136761,
        'index': 0,
        'valid_loss': 3730.2055448889732,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.6812251843448667,
        'rec': 0.7492202121023082,
        'f1': 0.7136066547831253,
        'bacc': 0.7918169416433756,
        'index': 1,
        'valid_loss': 2465.1887937039137,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7665271966527196,
        'rec': 0.5714285714285714,
        'f1': 0.6547533952823446,
        'bacc': 0.7446123411061537,
        'index': 0,
        'valid_loss': 3434.219853579998,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7661469933184856,
        'rec': 0.6437928883343731,
        'f1': 0.6996610169491525,
        'bacc': 0.7754910228354246,
        'index': 1,
        'valid_loss': 3845.580547273159,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7628205128205128,
        'rec': 0.6681222707423581,
        'f1': 0.712337878284004,
        'bacc': 0.7850039756776022,
        'index': 1,
        'valid_loss': 2416.555756121874,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.798202614379085,
        'rec': 0.6094822208359326,
        'f1': 0.6911920764060842,
        'bacc': 0.7683533673419498,
        'index': 1,
        'valid_loss': 3349.0502215623856,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7444682369735903,
        'rec': 0.6506550218340611,
        'f1': 0.6944074567243675,
        'bacc': 0.7725873812764884,
        'index': 0,
        'valid_loss': 3837.9747716784477,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7301899148657498,
        'rec': 0.6955708047411104,
        'f1': 0.7124600638977635,
        'bacc': 0.7870900576445682,
        'index': 1,
        'valid_loss': 2468.254655316472,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.782808902532617,
        'rec': 0.6363069245165315,
        'f1': 0.7019958706125259,
        'bacc': 0.7764622424586193,
        'index': 1,
        'valid_loss': 3415.44566655159,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7445887445887446,
        'rec': 0.6437928883343731,
        'f1': 0.6905319504851121,
        'bacc': 0.7697455897181589,
        'index': 0,
        'valid_loss': 3803.2184776067734,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7625739644970414,
        'rec': 0.6431690580162196,
        'f1': 0.6978003384094755,
        'bacc': 0.7742951948890762,
        'index': 1,
        'valid_loss': 2444.617573261261,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.6949055523755009,
        'rec': 0.7573300062383032,
        'f1': 0.7247761194029851,
        'bacc': 0.8001440838498528,
        'index': 1,
        'valid_loss': 3300.1676201820374,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7652439024390244,
        'rec': 0.6263256394260761,
        'f1': 0.6888507718696398,
        'bacc': 0.7677886299664264,
        'index': 0,
        'valid_loss': 3864.5796389579773,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7601683029453016,
        'rec': 0.6762320648783531,
        'f1': 0.7157477715417631,
        'bacc': 0.7877330035646921,
        'index': 1,
        'valid_loss': 2484.199497073889,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7220405493786789,
        'rec': 0.6887086712414223,
        'f1': 0.7049808429118773,
        'bacc': 0.7817438465223022,
        'index': 1,
        'valid_loss': 3390.7763091623783,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7394317394317395,
        'rec': 0.6656269494697442,
        'f1': 0.7005909389363099,
        'bacc': 0.777421606732515,
        'index': 0,
        'valid_loss': 3662.0192694664,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7892690513219285,
        'rec': 0.6331877729257642,
        'f1': 0.7026652821045345,
        'bacc': 0.776670492237779,
        'index': 1,
        'valid_loss': 2477.4486769735813,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7501789549033644,
        'rec': 0.6537741734248285,
        'f1': 0.6986666666666665,
        'bacc': 0.7754728262527796,
        'index': 0,
        'valid_loss': 3346.9909251332283,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7258785942492013,
        'rec': 0.7086712414223332,
        'f1': 0.7171717171717172,
        'bacc': 0.7911358564212432,
        'index': 0,
        'valid_loss': 3703.2602721452713,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7112462006079028,
        'rec': 0.7298814722395508,
        'f1': 0.7204433497536945,
        'bacc': 0.7949643071274359,
        'index': 0,
        'valid_loss': 2524.4548586905003,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7415584415584415,
        'rec': 0.7124142233312539,
        'f1': 0.7266942411708558,
        'bacc': 0.7975742301099404,
        'index': 2,
        'valid_loss': 3369.7071203291416,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7734627831715211,
        'rec': 0.5963817841547099,
        'f1': 0.6734765762592462,
        'bacc': 0.7569416286713444,
        'index': 0,
        'valid_loss': 3734.2472842931747,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7687943262411348,
        'rec': 0.6762320648783531,
        'f1': 0.7195486226352473,
        'bacc': 0.7900901043307499,
        'index': 1,
        'valid_loss': 2404.9182975292206,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7297297297297297,
        'rec': 0.7074235807860262,
        'f1': 0.7184035476718403,
        'bacc': 0.7918378952839972,
        'index': 1,
        'valid_loss': 3297.1612710654736,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7134292565947242,
        'rec': 0.74235807860262,
        'f1': 0.7276062366248853,
        'bacc': 0.8007606539153348,
        'index': 1,
        'valid_loss': 3773.286934375763,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7414266117969822,
        'rec': 0.6743605739238927,
        'f1': 0.7063051290427964,
        'bacc': 0.7816411001617106,
        'index': 1,
        'valid_loss': 2384.9628913998604,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.750351617440225,
        'rec': 0.6656269494697442,
        'f1': 0.7054545454545456,
        'bacc': 0.7805153014879658,
        'index': 1,
        'valid_loss': 3505.9041863679886,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7792721518987342,
        'rec': 0.6144728633811604,
        'f1': 0.6871294035577258,
        'bacc': 0.7661344870824482,
        'index': 0,
        'valid_loss': 3778.652960062027,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7098540145985401,
        'rec': 0.7280099812850904,
        'f1': 0.7188173698798891,
        'bacc': 0.7937339240544485,
        'index': 1,
        'valid_loss': 2479.055674403906,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.775,
        'rec': 0.6381784154709919,
        'f1': 0.6999657885733835,
        'bacc': 0.7753355247655489,
        'index': 1,
        'valid_loss': 3411.069749623537,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7255541069100391,
        'rec': 0.6943231441048034,
        'f1': 0.7095951546063117,
        'bacc': 0.7851403581455072,
        'index': 0,
        'valid_loss': 3793.9883954524994,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.8094059405940595,
        'rec': 0.6119775421085465,
        'f1': 0.6969804618117229,
        'bacc': 0.7719581287443145,
        'index': 1,
        'valid_loss': 2421.0949541926384,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7623076923076924,
        'rec': 0.6182158452900811,
        'f1': 0.6827419910437479,
        'bacc': 0.7635864141005503,
        'index': 0,
        'valid_loss': 3308.684137940407,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7102744097000638,
        'rec': 0.6943231441048034,
        'f1': 0.7022082018927445,
        'bacc': 0.7802788378155131,
        'index': 1,
        'valid_loss': 3966.7380640506744,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7073317307692307,
        'rec': 0.734248284466625,
        'f1': 0.7205387205387205,
        'bacc': 0.7953798876664298,
        'index': 1,
        'valid_loss': 2434.7542586177588,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.749131341209173,
        'rec': 0.6724890829694323,
        'f1': 0.7087442472057857,
        'bacc': 0.7830624554505382,
        'index': 1,
        'valid_loss': 3336.1833396553993,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7489331436699858,
        'rec': 0.6568933250155957,
        'f1': 0.6999002991026919,
        'bacc': 0.7764431268566487,
        'index': 0,
        'valid_loss': 3740.592990934849,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7558718861209964,
        'rec': 0.662507797878977,
        'f1': 0.7061170212765957,
        'bacc': 0.7807235512671256,
        'index': 1,
        'valid_loss': 2416.265726774931,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7487955953200275,
        'rec': 0.678727386150967,
        'f1': 0.712041884816754,
        'bacc': 0.785592331849791,
        'index': 1,
        'valid_loss': 3198.0504778325558,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7790788446526151,
        'rec': 0.6225826575171554,
        'f1': 0.6920943134535368,
        'bacc': 0.7696001089589313,
        'index': 0,
        'valid_loss': 3708.0996579527855,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7592592592592593,
        'rec': 0.6650031191515907,
        'f1': 0.7090123046225474,
        'bacc': 0.7827078058928254,
        'index': 1,
        'valid_loss': 2517.7877538204193,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.6851960210649503,
        'rec': 0.7305053025577043,
        'f1': 0.7071256038647343,
        'bacc': 0.7859951380201604,
        'index': 0,
        'valid_loss': 3347.1329513192177,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7248743718592965,
        'rec': 0.7199001871490954,
        'f1': 0.722378716744914,
        'bacc': 0.7954244601037168,
        'index': 1,
        'valid_loss': 3937.4462536871433,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.727735368956743,
        'rec': 0.7136618839675608,
        'f1': 0.7206299212598425,
        'bacc': 0.7937784964917356,
        'index': 2,
        'valid_loss': 2472.847477763891,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7152605459057072,
        'rec': 0.7192763568309419,
        'f1': 0.7172628304821151,
        'bacc': 0.7920188501891893,
        'index': 1,
        'valid_loss': 3343.8586317002773,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7624350408314774,
        'rec': 0.6406737367436057,
        'f1': 0.696271186440678,
        'bacc': 0.7731948530506478,
        'index': 0,
        'valid_loss': 3686.3260063529015,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7491190979563073,
        'rec': 0.6631316281971303,
        'f1': 0.7035076108537393,
        'bacc': 0.7791203220537802,
        'index': 1,
        'valid_loss': 2403.5631217360497,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.6827927432655305,
        'rec': 0.7747972551466001,
        'f1': 0.7258912916423144,
        'bacc': 0.8023956811973425,
        'index': 0,
        'valid_loss': 3344.088317513466,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7598870056497176,
        'rec': 0.6712414223331253,
        'f1': 0.7128188141768798,
        'bacc': 0.7855323198878355,
        'index': 1,
        'valid_loss': 3864.0149224102497,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.80479735318445,
        'rec': 0.6069868995633187,
        'f1': 0.6920341394025604,
        'bacc': 0.7687262134823076,
        'index': 1,
        'valid_loss': 2540.6422351002693,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7479281767955801,
        'rec': 0.6756082345601996,
        'f1': 0.7099311701081613,
        'bacc': 0.7840327560544074,
        'index': 1,
        'valid_loss': 3253.8963209092617,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.6852976913730255,
        'rec': 0.7036805988771054,
        'f1': 0.6943674976915974,
        'bacc': 0.7755291621374332,
        'index': 0,
        'valid_loss': 3823.136471390724,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7237604636188023,
        'rec': 0.7011852776044916,
        'f1': 0.7122940430925222,
        'bacc': 0.7873928745123224,
        'index': 1,
        'valid_loss': 2380.0733022987843,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7077016373559734,
        'rec': 0.7280099812850904,
        'f1': 0.7177121771217713,
        'bacc': 0.7929973300650555,
        'index': 2,
        'valid_loss': 3449.5297749638557,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.8423280423280424,
        'rec': 0.49656893325015594,
        'f1': 0.6248037676609105,
        'bacc': 0.7263339657411652,
        'index': 0,
        'valid_loss': 3996.6766325235367,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7540069686411149,
        'rec': 0.6749844042420462,
        'f1': 0.7123107307439105,
        'bacc': 0.785488666469874,
        'index': 0,
        'valid_loss': 2350.4450199604034,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7537050105857446,
        'rec': 0.6662507797878977,
        'f1': 0.7072847682119204,
        'bacc': 0.7817111294343142,
        'index': 1,
        'valid_loss': 3346.454794704914,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7029940119760479,
        'rec': 0.7323767935121647,
        'f1': 0.7173846623892454,
        'bacc': 0.7931182730082922,
        'index': 0,
        'valid_loss': 3762.0082406401634,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.8077246011754828,
        'rec': 0.6001247660636307,
        'f1': 0.6886184681460273,
        'bacc': 0.7663263783176139,
        'index': 1,
        'valid_loss': 2503.4904802143574,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7461069735951252,
        'rec': 0.6874610106051154,
        'f1': 0.7155844155844155,
        'bacc': 0.7884859560980793,
        'index': 1,
        'valid_loss': 3400.921870291233,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7978533094812165,
        'rec': 0.5564566437928883,
        'f1': 0.6556413083425211,
        'bacc': 0.7449342735758784,
        'index': 0,
        'valid_loss': 3786.4604108929634,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7592458303118201,
        'rec': 0.653150343106675,
        'f1': 0.7022132796780683,
        'bacc': 0.7776653306576392,
        'index': 1,
        'valid_loss': 2449.0482515990734,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7694566813509545,
        'rec': 0.6537741734248285,
        'f1': 0.7069139966273188,
        'bacc': 0.7806289841785309,
        'index': 0,
        'valid_loss': 3401.431262552738,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.730188679245283,
        'rec': 0.7242669993761697,
        'f1': 0.7272157845286564,
        'bacc': 0.7989337353981614,
        'index': 1,
        'valid_loss': 3744.5027616024017,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7043795620437956,
        'rec': 0.7223955084217093,
        'f1': 0.7132737911918695,
        'bacc': 0.7896008184418506,
        'index': 1,
        'valid_loss': 2484.9046314656734,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7987012987012987,
        'rec': 0.6138490330630069,
        'f1': 0.6941798941798942,
        'bacc': 0.7703894546576083,
        'index': 0,
        'valid_loss': 3379.3624133467674,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7741211667913238,
        'rec': 0.6456643792888335,
        'f1': 0.7040816326530612,
        'bacc': 0.7783419126850768,
        'index': 1,
        'valid_loss': 3860.5444750785828,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    {
        'prec': 0.7305389221556886,
        'rec': 0.6849656893325016,
        'f1': 0.7070186735350934,
        'bacc': 0.782818731525414,
        'index': 1,
        'valid_loss': 2429.7412655353546,
        'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    },
    {
        'prec': 0.7590940288263556,
        'rec': 0.6899563318777293,
        'f1': 0.722875816993464,
        'bacc': 0.7932692678834727,
        'index': 1,
        'valid_loss': 3359.0035759210587,
        'desc': 'Sector_Split 2vs2 rate=0.3 early_stop'
    },
    {
        'prec': 0.7485955056179775,
        'rec': 0.6650031191515907,
        'f1': 0.7043277172117608,
        'bacc': 0.7797614299352532,
        'index': 1,
        'valid_loss': 3723.9444106817245,
        'desc': 'Sector_Split 2vs2 rate=0.4 early_stop'
    },
    # {
    #     'prec': 0.7545327754532776,
    #     'rec': 0.6749844042420462,
    #     'f1': 0.7125452749423774,
    #     'bacc': 0.7856359852677526,
    #     'index': 1,
    #     'valid_loss': 2424.8027932345867,
    #     'desc': 'Sector_Split 2vs2 rate=0.1 early_stop'
    # }
]

standard_1vs1 = [{
    'prec':
    0.7127659574468085,
    'rec':
    0.6265586034912718,
    'f1':
    0.6668878566688784,
    'bacc':
    0.7536327627176536,
    'index':
    1,
    'valid_loss':
    2289.3229060173035,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.698961937716263,
    'rec':
    0.6296758104738155,
    'f1':
    0.6625122991144639,
    'bacc':
    0.7507731040587045,
    'index':
    1,
    'valid_loss':
    2254.3757667839527,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7045296167247387,
    'rec':
    0.6302992518703242,
    'f1':
    0.6653504442250741,
    'bacc':
    0.7527048542120398,
    'index':
    1,
    'valid_loss':
    2343.0792006254196,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7599225556631172,
    'rec':
    0.48940149625935164,
    'f1':
    0.5953735305271142,
    'bacc':
    0.7081764476878496,
    'index':
    0,
    'valid_loss':
    2284.975489228964,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.6490299823633157,
    'rec':
    0.6882793017456359,
    'f1':
    0.6680786686838124,
    'bacc':
    0.7562162340834218,
    'index':
    0,
    'valid_loss':
    2372.5224201083183,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.6742751388032079,
    'rec':
    0.6814214463840399,
    'f1':
    0.6778294573643411,
    'bacc':
    0.7629493093481319,
    'index':
    1,
    'valid_loss':
    2335.3313271850348,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7528089887640449,
    'rec':
    0.5847880299251871,
    'f1':
    0.6582456140350876,
    'bacc':
    0.7470331902203255,
    'index':
    1,
    'valid_loss':
    2267.5584505349398,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.69140625,
    'rec':
    0.6620947630922693,
    'f1':
    0.6764331210191084,
    'bacc':
    0.7612388395726443,
    'index':
    1,
    'valid_loss':
    2203.007606089115,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.6919365954514128,
    'rec':
    0.6259351620947631,
    'f1':
    0.6572831423895255,
    'bacc':
    0.7471354750090899,
    'index':
    1,
    'valid_loss':
    2255.3536217808723,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7132075471698113,
    'rec':
    0.5891521197007481,
    'f1':
    0.6452714236940936,
    'bacc':
    0.7386114059475758,
    'index':
    0,
    'valid_loss':
    2288.7229233384132,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7392325763508223,
    'rec':
    0.5885286783042394,
    'f1':
    0.6553280111072545,
    'bacc':
    0.7452216292846676,
    'index':
    1,
    'valid_loss':
    2362.817992478609,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.790547798066595,
    'rec':
    0.45885286783042395,
    'f1':
    0.5806706114398422,
    'bacc':
    0.7007077299387761,
    'index':
    0,
    'valid_loss':
    2311.865593135357,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7096774193548387,
    'rec':
    0.5897755610972568,
    'f1':
    0.6441947565543071,
    'bacc':
    0.7378921988107786,
    'index':
    0,
    'valid_loss':
    2275.7051709890366,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.6929936305732484,
    'rec':
    0.6783042394014963,
    'f1':
    0.6855702583490864,
    'bacc':
    0.7681653744871988,
    'index':
    1,
    'valid_loss':
    2283.6767383515835,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7266187050359713,
    'rec':
    0.566708229426434,
    'f1':
    0.6367775831873905,
    'bacc':
    0.7329859262006986,
    'index':
    1,
    'valid_loss':
    2265.253756046295,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7609942638623327,
    'rec':
    0.49625935162094764,
    'f1':
    0.600754716981132,
    'bacc':
    0.7113108245586329,
    'index':
    0,
    'valid_loss':
    2315.254867851734,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.6839972991222147,
    'rec':
    0.6315461346633416,
    'f1':
    0.6567260940032416,
    'bacc':
    0.7468481777882245,
    'index':
    1,
    'valid_loss':
    2280.258800506592,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7859154929577464,
    'rec':
    0.5218204488778054,
    'f1':
    0.6272011989509179,
    'bacc':
    0.7273314320972237,
    'index':
    1,
    'valid_loss':
    2272.8926632106304,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.764265668849392,
    'rec':
    0.5093516209476309,
    'f1':
    0.6112981668537224,
    'bacc':
    0.7175624084119598,
    'index':
    0,
    'valid_loss':
    2251.3199159801006,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}, {
    'prec':
    0.7430497051390059,
    'rec':
    0.5498753117206983,
    'f1':
    0.6320315299175923,
    'bacc':
    0.7300186573331031,
    'index':
    0,
    'valid_loss':
    2254.3747940063477,
    'desc':
    'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop'
}]

sector_early_3vs3 = [{
    'prec': 0.7841781874039938,
    'rec': 0.636930754834685,
    'f1': 0.7029259896729776,
    'bacc': 0.7770443868513047,
    'index': 1,
    'valid_loss': 3733.684864576906,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7389033942558747,
    'rec': 0.7061759201497193,
    'f1': 0.7221690590111642,
    'bacc': 0.7941256959239162,
    'index': 1,
    'valid_loss': 3769.8631914779544,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7169117647058824,
    'rec': 0.7298814722395508,
    'f1': 0.7233384853168471,
    'bacc': 0.7968393210254358,
    'index': 1,
    'valid_loss': 3846.3191336989403,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.737183646982479,
    'rec': 0.7086712414223332,
    'f1': 0.7226463104325701,
    'bacc': 0.7946363282583364,
    'index': 0,
    'valid_loss': 3620.837383583188,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7496551724137931,
    'rec': 0.6781035558328135,
    'f1': 0.7120864723223059,
    'bacc': 0.7855435231994257,
    'index': 1,
    'valid_loss': 3855.3617739900947,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7274939172749392,
    'rec': 0.7461010605115409,
    'f1': 0.7366800123190639,
    'bacc': 0.8070127944067138,
    'index': 1,
    'valid_loss': 3746.2740967422724,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7348284960422163,
    'rec': 0.6949469744229569,
    'f1': 0.7143315165117025,
    'bacc': 0.7882164117397803,
    'index': 1,
    'valid_loss': 3725.844294860959,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7420446851726472,
    'rec': 0.6837180286961946,
    'f1': 0.7116883116883116,
    'bacc': 0.7856974577443236,
    'index': 1,
    'valid_loss': 3903.038856871426,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7803425167535368,
    'rec': 0.6537741734248285,
    'f1': 0.7114731839782756,
    'bacc': 0.7834024169010935,
    'index': 1,
    'valid_loss': 3830.312257371843,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7416612164813604,
    'rec': 0.7074235807860262,
    'f1': 0.7241379310344828,
    'bacc': 0.7954865545439564,
    'index': 0,
    'valid_loss': 3730.653361015022,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7700573065902578,
    'rec': 0.670617592014972,
    'f1': 0.7169056352117372,
    'bacc': 0.7879915790263539,
    'index': 1,
    'valid_loss': 3640.137948781252,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7462987886944819,
    'rec': 0.6918278228321897,
    'f1': 0.7180317254775009,
    'bacc': 0.7903419774538307,
    'index': 1,
    'valid_loss': 3621.2846271693707,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7361647361647362,
    'rec': 0.7136618839675608,
    'f1': 0.7247386759581882,
    'bacc': 0.7963946212290633,
    'index': 0,
    'valid_loss': 3620.860544934869,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7603530210454854,
    'rec': 0.6986899563318777,
    'f1': 0.7282184655396619,
    'bacc': 0.7973107800527313,
    'index': 0,
    'valid_loss': 3730.729187257588,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7001743172574085,
    'rec': 0.751715533374922,
    'f1': 0.7250300842358605,
    'bacc': 0.799796445932744,
    'index': 1,
    'valid_loss': 3727.7601437047124,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7382864792503346,
    'rec': 0.6880848409232688,
    'f1': 0.7123022279625444,
    'bacc': 0.7864068072540873,
    'index': 1,
    'valid_loss': 3643.334842979908,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.783608914450036,
    'rec': 0.6799750467872738,
    'f1': 0.7281229124916501,
    'bacc': 0.795618419620052,
    'index': 1,
    'valid_loss': 3675.029540270567,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7223241590214067,
    'rec': 0.7367436057392389,
    'f1': 0.7294626312538605,
    'bacc': 0.8014496330582987,
    'index': 1,
    'valid_loss': 3718.416021153331,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.7339274347549332,
    'rec': 0.7192763568309419,
    'f1': 0.7265280403276623,
    'bacc': 0.7980226123777352,
    'index': 1,
    'valid_loss': 3832.8079026937485,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}, {
    'prec': 0.6556219445953286,
    'rec': 0.7529631940112289,
    'f1': 0.7009291521486644,
    'bacc': 0.7830264083263692,
    'index': 0,
    'valid_loss': 3928.52016890049,
    'desc': 'Sector_Split 3vs3 rate=0.2 early_stop'
}]
