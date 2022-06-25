## dd
# train = load_dataset('conll2003', split = 'train')

# path: /usr01/ZhuoBinggang/enwiki_20180420_win10_300d.pkl
# MODEL_FILE = '/usr01/ZhuoBinggang/enwiki_20180420_win10_300d.pkl'
# wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
# emb = wiki2vec.get_word_vector(word)
import torch as t



# (0,2) -> (1,0)

# numerator
def cal_numerators(datas):
    prev_state = 2
    numerators = []
    for x, y in datas:
        t_score = trans[prev_state, x]
        e_score = emiss[x, y]
        numerators.append(t_score * e_score)
        prev_state = x
    return numerators

# 错误的实现基于一个假设(以下两式相等)： 
# (x1 + x2) * (x11 + x12 + x21 + x22)
# (x1 * x11) + (x1 * x12) + (x2 * x21) + (x2 * x22)
def cal_denominators_fault(datas):
    BOS = 2
    n_states = 2
    denominators = []
    denominator_0 = 0
    for state in range(n_states):
        t_score = trans[BOS, state] 
        e_scores = sum(emiss[state, :])
        denominator_0 += (t_score * e_scores)
    denominators.append(denominator_0)
    for i in range(1, len(datas)):
        denominator = 0
        for prev_state in range(n_states):
            for cur_state in range(n_states):
                t_score = trans[prev_state, cur_state]
                e_scores = sum(emiss[cur_state, :])
                denominator += (t_score * e_scores)
        denominators.append(denominator)
    return denominators


def cal_denominators(datas):
    BOS = 2
    n_states = 2
    denominators = []
    prev_denominator = []
    for state in range(n_states):
        t_score = trans[BOS, state] 
        e_scores = sum(emiss[state, :]) 
        prev_denominator.append(t_score * e_scores)
    denominators.append(prev_denominator)
    for i in range(1, len(datas)):
        denominator = []
        for cur_state in range(n_states):
            denominator_cur_state = 0
            for prev_state in range(n_states):
                t_score = trans[prev_state, cur_state]
                e_scores = sum(emiss[cur_state, :])
                # print(len(prev_denominator))
                denominator_cur_state += t_score * e_scores * prev_denominator[prev_state]
            denominator.append(denominator_cur_state)
        prev_denominator = denominator
        denominators.append(denominator)
    return  prev_denominator, denominators


emiss = t.tensor([[6,2,2],[2,4,4]])
trans = t.tensor([[8,2],[6,4],[8,2]])
# datas = [(0, 2), (1, 0)]
datas = [(0, 2), (1, 0), (0, 1)]

# 一个例子
def run():
    nom = cal_numerators(datas)
    denom, _ = cal_denominators(datas)
    result = (nom[0] * nom[1] * nom[2]) / sum(denom)
    print(result)




