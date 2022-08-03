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


def create_iteration_callback_baseline(key, m, ld_dev, intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    count = 0
    def record():
        print('record')
        prec, rec, f, _ = test_chain_baseline(m, ld_dev)
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


def create_iteration_callback(key, m, ld_dev, intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    count = 0
    def record():
        print('record')
        prec, rec, f, _ = test_chain(m, ld_dev)
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

def train_and_plot_by_iteration_ours():
    times = 3
    epochs = 3
    ld_train = read_ld_train()
    ld_dev = read_ld_dev() 
    # 5 * (2 + 2 + 2 + 1) * 25 = 875(min) = 14.58(hour)
    for model_idx in range(times):
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback(f'AUXFL_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1, iteration_callback = cb)
    for model_idx in range(times):
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback(f'AUX_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train(m, ld_train, fl_rate = 0, aux_rate = 0.3, iteration_callback = cb)


def train_and_plot_by_iteration_theirs(kind = 0):
    times = 3
    epochs = 3
    ld_train = read_ld_train()
    ld_dev = read_ld_dev() 
    # 5 * (2 + 2 + 2 + 1) * 25 = 875(min) = 14.58(hour)
    for model_idx in range(times):
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'FL_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 5.0, iteration_callback = cb)
    for model_idx in range(times):
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'STAND_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 0, iteration_callback = cb)


def expand_scale(fs):
    res = []
    for f in fs[:50]: # 0 ~ 49 => 10 ~ 500
        res.append(f)
    for f in fs[50:]: # 50 ~ 97 => 500 ~ 5200
        res += [f] * 10
    return res

def labels():
    start = 10
    itensive = []
    for i in range(0, 50):
        itensive.append(start + i * 10)
    start = 600
    normal = []
    for i in range(0, 47):
        normal.append(start + i * 100)
    return itensive + normal

def labels2(intensively_log_interval = 10, intensive_log_until = 500, normal_log_interval = 100):
    res = []
    for i in range(1, 5201):
        if i < intensive_log_until:
            if i % intensively_log_interval == 0:
                res.append(i)
        else:
            if i % normal_log_interval == 0:
                res.append(i)
    return res







