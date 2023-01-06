from exp_news import *

# tests = read_ld_tests()
# news_test = tests[3]
testds = read_combined_test()

# SEED = 19
# SEED = 666
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

def get_case2(labels, y1, y2, testset):
    res = []
    for label, auxfl, fl, case in zip(labels, y1, y2, testset):
        if label == 1 and auxfl < 0.5 and fl > 0.5:
            res.append((fl - 0.5 + 0.5 - auxfl, auxfl, fl, case))
    return res

