from exp_novel import *

novel_test = read_ld_test_from_chapters()

# SEED = 19
# SEED = 666
def get_results(testset, seed, index):
    m = load_model(f'SEED_{seed}_AUX01FL50E2_{index}')
    y_true1, y_pred1 = test(testset, m)
    m = load_model(f'SEED_{seed}_FL50E2_{index}')
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

# labels, y1, _, y2 = get_results(novel_test, 14)
