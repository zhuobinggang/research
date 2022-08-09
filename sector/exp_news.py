from novel_learning_curve import *
from mainichi_paragraph import read_ld_train, read_ld_tests, read_ld_test, read_ld_dev

SEEDS_FOR_TRAIN = [21, 22, 8, 4, 14, 3, 19, 97, 10, 666]

def save_model(m, name):
    t.save(m, f'./save/{name}.tch')

def create_model_with_seed(seed):
    t.manual_seed(seed)
    m = Sector_2022()
    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'created model with seed {seed} at time {time_string}')
    return m

def train_and_save(start = 0, times = 10):
    ld_train = read_ld_train()
    for model_idx_org in range(times):
        model_idx = model_idx_org + start
        SEED = SEEDS_FOR_TRAIN[model_idx]
        m = create_model_with_seed(SEED)
        for i in range(2): # AUX_FL
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
        save_model(m, f'SEED{SEED}_AUX01FL50E2.tch')
        m = create_model_with_seed(SEED)
        for i in range(2): # AUX
            train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        save_model(m, f'SEED{SEED}_AUX02E2.tch')
        m = create_model_with_seed(SEED)
        for i in range(3): # FL
            train_baseline(m, ld_train, fl_rate = 2.0)
        save_model(m, f'SEED{SEED}_FL20E3.tch')
        m = create_model_with_seed(SEED)
        for i in range(2): # STAND
            train_baseline(m, ld_train, fl_rate = 0)
        save_model(m, f'SEED{SEED}_STDE3.tch')


# NOTE: 新闻数据集的最佳参数和小说不一样！
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
            train(m, ld_train, fl_rate = 0, aux_rate = 0.2, iteration_callback = cb)
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'FL_FS_DEV{model_idx}', m, ld_dev, intensively_log_interval = 20)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 2.0, iteration_callback = cb)
        t.manual_seed(RANDOM_SEEDs[model_idx])
        m = Sector_2022()
        cb = create_iteration_callback_baseline(f'STAND_FS_DEV{model_idx}', m, ld_dev, intensively_log_interval = 20)
        for i in range(epochs):
            train_baseline(m, ld_train, fl_rate = 0, iteration_callback = cb)


