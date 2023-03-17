from exp_novel import *

SEEDS_FOR_TRAIN = [20, 22, 8, 4, 13, 3, 19, 97, 10, 666, 21, 14, 555]
# SEEDS_FOR_TRAIN2 = [22, 8, 4, 3, 19, 97, 10, 666, 21, 14]
# SEED_FOR_RANDOM = 31 
SEED_FOR_RANDOM = 32 # 人手实验

# 获取100个随机下标，范围 = (0, 11098)
def random_100_index():
    # np.random.seed(seed=32)
    np.random.seed(seed=SEED_FOR_RANDOM)
    res = np.random.choice(range(11098), 100, replace=False)
    np.random.seed()
    return res

def select_from_dstest(dstest, idxs):
    mld = [dstest[idx] for idx in idxs]
    return mld

def run(times = 10, start = 0):
    dstest = read_ld_test()
    idxs = random_100_index()
    mld = select_from_dstest(dstest, idxs)
    ress = []
    for model_idx in range(times):
        model_idx = model_idx + start
        SEED = SEEDS_FOR_TRAIN[model_idx]
        m = load_model(f'SEED_{SEED}_AUX01FL50E2_{model_idx}')
        # m = load_model(f'SEED_{SEED}_FL50E2_{model_idx}')
        res = test_chain(m, mld)
        ress.append(res)
    return ress


# one_of_all = 3625 / 28178 (小説トレーニングセットの割合)
def random_one_by_trainset(size = 100, one_of_all = 0.5):
    res = []
    np.random.seed(seed=SEED_FOR_RANDOM)
    for i in range(size):
        if np.random.rand() < one_of_all:
            res.append(1)
        else:
            res.append(0)
    return res





