from novel_learning_curve import *
from mainichi_paragraph import read_ld_train, read_ld_tests, read_ld_test, read_ld_dev

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
