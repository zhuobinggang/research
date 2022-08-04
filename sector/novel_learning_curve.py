from sector import *
from dataset_for_sector import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters, read_lds_test_from_chapters

read_ld_train = read_ld_train_from_chapters
read_ld_test = read_ld_test_from_chapters
read_ld_dev = read_ld_dev_from_chapters

RANDOM_SEEDs = [2022, 2023, 2024]

dic = {
    'STAND_FS_TEST0': [],
    'STAND_FS_TEST1': [],
    'STAND_FS_TEST2': [],
    'STAND_FS_DEV0': [],
    'STAND_FS_DEV1': [],
    'STAND_FS_DEV2': [],
    'FL_FS_TEST0': [],
    'FL_FS_TEST1': [],
    'FL_FS_TEST2': [],
    'FL_FS_DEV0': [],
    'FL_FS_DEV1': [],
    'FL_FS_DEV2': [],
    'AUX_FS_TEST0': [],
    'AUX_FS_TEST1': [],
    'AUX_FS_TEST2': [],
    'AUX_FS_DEV0': [],
    'AUX_FS_DEV1': [],
    'AUX_FS_DEV2': [],
    'AUXFL_FS_TEST0': [],
    'AUXFL_FS_TEST1': [],
    'AUXFL_FS_TEST2': [],
    'AUXFL_FS_DEV0': [],
    'AUXFL_FS_DEV1': [],
    'AUXFL_FS_DEV2': [],
}

def save_dic(name = 'exp_novel.txt'):
    f = open(name, 'w')
    f.write(str(dic))
    f.close()

def create_iteration_callback_shell(key, m, ld_dev, test_function, intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    count = 0
    def record():
        print('record')
        prec, rec, f, _ = test_function(m, ld_dev)
        dic[key].append(f)
        save_dic()
    def cb():
        nonlocal count
        count += 1
        if count < intensive_log_until:
            if count % intensively_log_interval == 0:
                record()
        else:
            if count % normal_log_interval == 0:
                record()
    return cb

def create_iteration_callback_baseline(key, m, ld_dev, intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    return create_iteration_callback_shell(key, m, ld_dev, test_chain_baseline, intensively_log_interval, intensive_log_until, normal_log_interval)


def create_iteration_callback(key, m, ld_dev, intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    return create_iteration_callback_shell(key, m, ld_dev, test_chain, intensively_log_interval, intensive_log_until, normal_log_interval)


def labels(intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    res = []
    for i in range(1, 5201):
        if i < intensive_log_until:
            if i % intensively_log_interval == 0:
                res.append(i)
        else:
            if i % normal_log_interval == 0:
                res.append(i)
    return res


# NOTE: 新闻数据集的最佳参数和小说不一样！
# 根据第二次grid search的结果，两次参数几乎一样，保留，但是epoch不一样
def train_and_plot(times = 3, start = 0):
    epochs = 3
    ld_train = read_ld_train()
    ld_dev = read_ld_dev() 
    for model_idx_org in range(times):
        model_idx = model_idx_org + start
        print(f'random seed {RANDOM_SEEDs[model_idx]}')
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback(f'AUXFL_FS_DEV{model_idx}', m, ld_dev, intensively_log_interval = 20)
        for i in range(epochs):
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1, iteration_callback = cb)
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback(f'AUX_FS_DEV{model_idx}', m, ld_dev, intensively_log_interval = 20)
        for i in range(epochs):
            train(m, ld_train, fl_rate = 0, aux_rate = 0.3, iteration_callback = cb)
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'FL_FS_DEV{model_idx}', m, ld_dev, intensively_log_interval = 20)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 5.0, iteration_callback = cb)
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'STAND_FS_DEV{model_idx}', m, ld_dev, intensively_log_interval = 20)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 0, iteration_callback = cb)




