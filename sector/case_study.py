from exp_news import *
from case_study_novel import view_att, rank, rank_by_p, filter_special_tokens

# tests = read_ld_tests()
# news_test = tests[3]
testds = read_combined_test()

# seed = 97
def get_results(testset, seed):
    m = load_model(f'SEED{seed}_AUX01FL50E2')
    y_true1, y_pred1 = test(testset, m)
    m = load_model(f'SEED{seed}_FL20E3')
    y_true2, y_pred2 = test(testset, m)
    return y_true1, y_pred1, y_true2, y_pred2

def get_case1(labels, y1, y2, testset):
    res = []
    for label, auxfl, fl, case in zip(labels, y1, y2, testset):
        if label == 1 and auxfl > 0.5 and fl < 0.5:
            res.append((auxfl - 0.5 + 0.5 - fl, auxfl, fl, case))
    return res
# selected index: 1

def get_case2(labels, y1, y2, testset):
    res = []
    for label, auxfl, fl, case in zip(labels, y1, y2, testset):
        if label == 1 and auxfl < 0.5 and fl > 0.5:
            res.append((fl - 0.5 + 0.5 - auxfl, auxfl, fl, case))
    return res

### 

def run():
    seed = 97
    m_aux = load_model(f'SEED{seed}_AUX01FL50E2')
    m_fl = load_model(f'SEED{seed}_FL20E3')
    return rank(m_aux, testds, stand = False), rank(m_fl, testds, stand = True)

