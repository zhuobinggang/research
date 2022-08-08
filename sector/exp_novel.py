from sector import *
from datetime import datetime
from dataset_for_sector import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters, read_lds_test_from_chapters

read_ld_train = read_ld_train_from_chapters
read_ld_test = read_ld_test_from_chapters
read_ld_dev = read_ld_dev_from_chapters

GRID_SEARCH_SEED = 777 # totally random

# NOTE: 20跟13导致AUX收束失败输出，将这两个替换掉 
SEEDS_FOR_TRAIN = [20, 22, 8, 4, 13, 3, 19, 97, 10, 666, 21, 14, 555]

def create_model_with_seed(seed):
    t.manual_seed(seed)
    m = Sector_2022()
    time_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'created model with seed {seed} at time {time_string}')
    return m

dic = {
  # AUX ONLY
  'E1FL0AUX00': [],        
  'E2FL0AUX00': [],       
  'E3FL0AUX00': [],       
  'E4FL0AUX00': [],       
  'E5FL0AUX00': [],       
  'E1FL0AUX01': [],       
  'E2FL0AUX01': [],     
  'E3FL0AUX01': [], 
  'E4FL0AUX01': [], 
  'E5FL0AUX01': [], 
  'E1FL0AUX02': [],       
  'E2FL0AUX02': [],      
  'E3FL0AUX02': [], 
  'E4FL0AUX02': [], 
  'E5FL0AUX02': [], 
  'E1FL0AUX03': [],       
  'E2FL0AUX03': [],     
  'E3FL0AUX03': [],
  'E4FL0AUX03': [],
  'E5FL0AUX03': [],

  # TODO: FL & AUX
  'E1FL05AUX00': [],        
  'E2FL05AUX00': [],       
  'E3FL05AUX00': [],       
  'E4FL05AUX00': [],       
  'E5FL05AUX00': [],       
  'E1FL10AUX00': [],       
  'E2FL10AUX00': [],     
  'E3FL10AUX00': [], 
  'E4FL10AUX00': [], 
  'E5FL10AUX00': [], 
  'E1FL20AUX00': [],       
  'E2FL20AUX00': [],      
  'E3FL20AUX00': [], 
  'E4FL20AUX00': [], 
  'E5FL20AUX00': [], 
  'E1FL50AUX00': [],       
  'E2FL50AUX00': [],     
  'E3FL50AUX00': [],
  'E4FL50AUX00': [],
  'E5FL50AUX00': [],

  'E1FL05AUX01': [],        
  'E2FL05AUX01': [],       
  'E3FL05AUX01': [],       
  'E4FL05AUX01': [],       
  'E5FL05AUX01': [],       
  'E1FL10AUX01': [],       
  'E2FL10AUX01': [],     
  'E3FL10AUX01': [], 
  'E4FL10AUX01': [], 
  'E5FL10AUX01': [], 
  'E1FL20AUX01': [],       
  'E2FL20AUX01': [],      
  'E3FL20AUX01': [], 
  'E4FL20AUX01': [], 
  'E5FL20AUX01': [], 
  'E1FL50AUX01': [],       
  'E2FL50AUX01': [],     
  'E3FL50AUX01': [],
  'E4FL50AUX01': [],
  'E5FL50AUX01': [],

  'E1FL05AUX02': [],        
  'E2FL05AUX02': [],       
  'E3FL05AUX02': [],       
  'E4FL05AUX02': [],       
  'E5FL05AUX02': [],       
  'E1FL10AUX02': [],       
  'E2FL10AUX02': [],     
  'E3FL10AUX02': [], 
  'E4FL10AUX02': [], 
  'E5FL10AUX02': [], 
  'E1FL20AUX02': [],       
  'E2FL20AUX02': [],      
  'E3FL20AUX02': [], 
  'E4FL20AUX02': [], 
  'E5FL20AUX02': [], 
  'E1FL50AUX02': [],       
  'E2FL50AUX02': [],     
  'E3FL50AUX02': [],
  'E4FL50AUX02': [],
  'E5FL50AUX02': [],

  'E1FL05AUX03': [],        
  'E2FL05AUX03': [],       
  'E3FL05AUX03': [],       
  'E4FL05AUX03': [],       
  'E5FL05AUX03': [],       
  'E1FL10AUX03': [],       
  'E2FL10AUX03': [],     
  'E3FL10AUX03': [], 
  'E4FL10AUX03': [], 
  'E5FL10AUX03': [], 
  'E1FL20AUX03': [],       
  'E2FL20AUX03': [],      
  'E3FL20AUX03': [], 
  'E4FL20AUX03': [], 
  'E5FL20AUX03': [], 
  'E1FL50AUX03': [],       
  'E2FL50AUX03': [],     
  'E3FL50AUX03': [],
  'E4FL50AUX03': [],
  'E5FL50AUX03': [],

  # FL ONLY
  'E1FL00': [], # BASELINE
  'E2FL00': [], # BASELINE
  'E3FL00': [], # BASELINE
  'E4FL00': [], # BASELINE
  'E5FL00': [], # BASELINE
  'E1FL05': [],
  'E2FL05': [],
  'E3FL05': [],
  'E4FL05': [],
  'E5FL05': [],
  'E1FL10': [],
  'E2FL10': [],
  'E3FL10': [],
  'E4FL10': [],
  'E5FL10': [],
  'E1FL20': [],
  'E2FL20': [],
  'E3FL20': [],
  'E4FL20': [],
  'E5FL20': [],
  'E1FL50': [],
  'E2FL50': [],
  'E3FL50': [],
  'E4FL50': [],
  'E5FL50': [],

  # compare
  'E1AUX': [],
  'E2AUX': [],
  'E3AUX': [],
  'E1FL': [],
  'E2FL': [],
  'E3FL': [],
  'E1AUX_FL': [],
  'E2AUX_FL': [],
  'E3AUX_FL': [],
  'E1STANDARD': [],
  'E2STANDARD': [],
  'E3STANDARD': [],

  # fs by chapters
  'AUX188': [], # (10, 188)
  'FL188': [], # (10, 188)
  'AUX_FL188': [], # (10, 188)
  'STAND188': [], # (10, 188)
}

def save_dic(name = 'exp_novel.txt'):
    f = open(name, 'w')
    f.write(str(dic))
    f.close()

def run_explore_AUX(seed):
    PATH = 'run_explore_AUX.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 2
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL0AUX00'
            train(m, ld_train, fl_rate = 0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL0AUX01'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL0AUX02'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL0AUX03'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL_AUX00(seed):
    PATH = 'run_explore_FL_AUX00.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 2
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL05AUX00'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL10AUX00'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL20AUX00'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL50AUX00'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL_AUX01(seed):
    PATH = 'run_explore_FL_AUX01.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 2
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL05AUX01'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL10AUX01'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL20AUX01'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL50AUX01'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL_AUX02(seed):
    PATH = 'run_explore_FL_AUX02.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 2
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL05AUX02'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL10AUX02'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL20AUX02'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL50AUX02'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)


def run_explore_FL_AUX03(seed):
    PATH = 'run_explore_FL_AUX03.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 2
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL05AUX03'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL10AUX03'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL20AUX03'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL50AUX03'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL(seed):
    PATH = 'run_explore_FL.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev()
    # 25 * 1 * 5 * 3 = 375 (mins) = 6.25 (hours)
    times = 1
    epochs = 2
    for _ in range(times): # Baseline BCE loss
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL00'
            train_baseline(m, ld_train, fl_rate = 0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL05'
            train_baseline(m, ld_train, fl_rate = 0.5)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL10'
            train_baseline(m, ld_train, fl_rate = 1.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL20'
            train_baseline(m, ld_train, fl_rate = 2.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = create_model_with_seed(seed)
        for i in range(epochs):
            key = f'E{i+1}FL50'
            train_baseline(m, ld_train, fl_rate = 5.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)


def run1(): # 16小时 -> 10小时 (改了epoch)
    run_explore_FL(GRID_SEARCH_SEED) # 6.25 hours
    run_explore_AUX(GRID_SEARCH_SEED) # 5 hours
    run_explore_FL_AUX00(GRID_SEARCH_SEED) # 5 hours


def run2(): # 15小时 -> 10小时
    run_explore_FL_AUX01(GRID_SEARCH_SEED) # 5 
    run_explore_FL_AUX02(GRID_SEARCH_SEED) # 5
    run_explore_FL_AUX03(GRID_SEARCH_SEED) # 5 

def save_model(m, name):
    t.save(m, f'/usr01/taku/sector_models/{name}.tch')

def load_model(name):
    print(f'load {name}')
    return t.load(f'/usr01/taku/sector_models/{name}.tch')

def train_and_save(start = 0, times = 10):
    # PATH = 'comparison.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_test() # NOTE: 必须是test
    # 10 * 25 * 2 * 4 = 2000 (min) = 33 (hours)
    for model_idx in range(times):
        model_idx = model_idx + start
        SEED = SEEDS_FOR_TRAIN[model_idx]
        m = create_model_with_seed(SEED)
        for i in range(2):
            key = f'E{i+1}AUX_FL'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
            # dic[key].append(test_chain(m, ld_test))
        save_model(m, f'SEED_{SEED}_AUX01FL50E2_{model_idx}')
        m = create_model_with_seed(SEED)
        for i in range(2):
            key = f'E{i+1}AUX'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
            # dic[key].append(test_chain(m, ld_test))
        save_model(m, f'SEED_{SEED}_AUX03E2_{model_idx}')
        m = create_model_with_seed(SEED)
        for i in range(2):
            key = f'E{i+1}FL'
            train_baseline(m, ld_train, fl_rate = 5.0)
            # dic[key].append(test_chain_baseline(m, ld_test))
        save_model(m, f'SEED_{SEED}_FL50E2_{model_idx}')
        m = create_model_with_seed(SEED)
        for i in range(2):
            key = f'E{i+1}STANDARD'
            train_baseline(m, ld_train, fl_rate = 0)
            # dic[key].append(test_chain_baseline(m, ld_test))
        save_model(m, f'SEED_{SEED}_STANDARDE2_{model_idx}')


def run_comparison_by_trained(start = 0,times = 10 ):
    PATH = 'comparisoned.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_test() # NOTE: 必须是test
    ld_dev = read_ld_dev_from_chapters() 
    # 5 * (2 + 2 + 2 + 1) * 25 = 875(min) = 14.58(hour)
    for model_idx in range(times):
        model_idx = model_idx + start
        SEED = SEEDS_FOR_TRAIN[model_idx]
        m = load_model(f'SEED_{SEED}_AUX01FL50E2_{model_idx}')
        dic['E1AUX_FL'].append(test_chain(m, ld_test))
        save_dic(PATH)
        m = load_model(f'SEED_{SEED}_AUX03E2_{model_idx}')
        dic['E1AUX'].append(test_chain(m, ld_test))
        save_dic(PATH)
        m = load_model(f'SEED_{SEED}_FL50E2_{model_idx}')
        dic['E1FL'].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
        m = load_model(f'SEED_{SEED}_STANDARDE2_{model_idx}')
        dic['E1STANDARD'].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)


def get_fs_by_lds_and_model(lds, m, test_function):
    fs = []
    for ld in lds:
        prec, rec, f, _ = test_function(m, ld)
        fs.append(f)
    return fs

def get_f_by_188_chapters(start = 0,times = 10 ):
    PATH = 'get_f_by_188_chapters.txt'
    ld_train = read_ld_train()
    lds = read_lds_test_from_chapters() # NOTE: 必须是test
    for model_idx in range(times):
        model_idx = model_idx + start
        SEED = SEEDS_FOR_TRAIN[model_idx]
        m = load_model(f'SEED_{SEED}_AUX01FL50E2_{model_idx}')
        dic['AUX_FL188'].append(get_fs_by_lds_and_model(lds, m, test_chain))
        save_dic(PATH)
        m = load_model(f'SEED_{SEED}_AUX03E2_{model_idx}')
        dic['AUX188'].append(get_fs_by_lds_and_model(lds, m, test_chain))
        save_dic(PATH)
        m = load_model(f'SEED_{SEED}_FL50E2_{model_idx}')
        dic['FL188'].append(get_fs_by_lds_and_model(lds, m, test_chain_baseline))
        save_dic(PATH)
        m = load_model(f'SEED_{SEED}_STANDARDE2_{model_idx}')
        dic['STAND188'].append(get_fs_by_lds_and_model(lds, m, test_chain_baseline))
        save_dic(PATH)


