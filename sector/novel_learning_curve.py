from sector import *
from dataset_for_sector import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters, read_lds_test_from_chapters

read_ld_train = read_ld_train_from_chapters
read_ld_test = read_ld_test_from_chapters
read_ld_dev = read_ld_dev_from_chapters

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


def create_iteration_callback_baseline(key, m, ld_dev):
    count = 0
    intensively_log_interval = 10
    intensive_log_until = 500
    normal_log_interval = 100
    def record():
        print('record')
        prec, rec, f, _ = test_chain_baseline(m, ld_dev)
        dic[key].append(f)
        save_dic()
    def cb():
        count += 1
        if count < intensive_log_until:
            if count % intensively_log_interval == 0:
                record()
        else:
            if count % normal_log_interval == 0:
                record()
    return cb


def create_iteration_callback(key, m, ld_dev):
    count = 0
    intensively_log_interval = 10
    intensive_log_until = 500
    normal_log_interval = 100
    def record():
        print('record')
        prec, rec, f, _ = test_chain(m, ld_dev)
        dic[key].append(f)
        save_dic()
    def cb():
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
        m = Sector_2022()
        cb = create_iteration_callback(f'AUXFL_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1, iteration_callback = cb)
    for model_idx in range(times):
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
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'FL_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 5.0, iteration_callback = cb)
    for model_idx in range(times):
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'STAND_FS_DEV{model_idx}', m, ld_dev)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 0, iteration_callback = cb)

