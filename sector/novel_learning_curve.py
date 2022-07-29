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
    def res():
        prec, rec, f, _ = test_chain_baseline(m, ld_dev)
        dic[key].append(f)
        save_dic()
    return res


def create_iteration_callback(key, m, ld_dev):
    def res():
        prec, rec, f, _ = test_chain(m, ld_dev)
        print('dd')
        dic[key].append(f)
    return res


def train_and_plot_by_iteration():
    PATH = 'comparison.txt'
    times = 1
    epochs = 1
    iteration = 100
    ld_train = read_ld_train()
    ld_test = read_ld_test() # NOTE: 必须是test
    ld_dev = read_ld_dev() 
    # 5 * (2 + 2 + 2 + 1) * 25 = 875(min) = 14.58(hour)
    for model_idx in range(times):
        m = Sector_2022()
        key = f'AUXFL_FS_DEV{model_idx}'
        cb = create_iteration_callback(key, m, ld_dev)
        for i in range(epochs):
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1, iteration_callback = cb)
    # for model_idx in range(times):
    #     m = Sector_2022()
    #     for i in range(1):
    #         key = f'E{i+1}AUX'
    #         train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
    #         # dic[key].append(test_chain(m, ld_test))
    #     save_dic(PATH)
    #     save_model(m, f'AUX03E1_{model_idx + 5}')
    # for model_idx in range(times):
    #     m = Sector_2022()
    #     for i in range(3):
    #         key = f'E{i+1}FL'
    #         train_baseline(m, ld_train, fl_rate = 5.0)
    #         # dic[key].append(test_chain_baseline(m, ld_test))
    #     save_dic(PATH)
    #     save_model(m, f'FL50E3_{model_idx + 5}')
    # for model_idx in range(times):
    #     m = Sector_2022()
    #     for i in range(2):
    #         key = f'E{i+1}STANDARD'
    #         train_baseline(m, ld_train, fl_rate = 0)
    #         # dic[key].append(test_chain_baseline(m, ld_test))
    #     save_dic(PATH)
    #     save_model(m, f'STANDARDE2_{model_idx + 5}')

