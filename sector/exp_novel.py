from sector import *
from dataset_for_sector import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters
read_ld_train = read_ld_train_from_chapters
read_ld_test = read_ld_test_from_chapters
read_ld_dev = read_ld_dev_from_chapters

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
  'E1FL': [],
  'E2FL': [],
  'E1AUX_FL': [],
  'E2AUX_FL': [],
  'E1STANDARD': [],
  'E2STANDARD': [],
}

def save_dic(name = 'exp_novel.txt'):
    f = open(name, 'w')
    f.write(str(dic))
    f.close()

def run_explore_AUX():
    PATH = 'run_explore_AUX.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 3
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL0AUX00'
            train(m, ld_train, fl_rate = 0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL0AUX01'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL0AUX02'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL0AUX03'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL_AUX00():
    PATH = 'run_explore_FL_AUX00.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 3
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL05AUX00'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL10AUX00'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL20AUX00'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL50AUX00'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL_AUX01():
    PATH = 'run_explore_FL_AUX01.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 3
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL05AUX01'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL10AUX01'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL20AUX01'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL50AUX01'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL_AUX02():
    PATH = 'run_explore_FL_AUX02.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 3
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL05AUX02'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL10AUX02'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL20AUX02'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL50AUX02'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.2)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)


def run_explore_FL_AUX03():
    PATH = 'run_explore_FL_AUX03.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 1 * 4 = 300 (mins) = 5 (hours)
    times = 1
    epochs = 3
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL05AUX03'
            train(m, ld_train, fl_rate = 0.5, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL10AUX03'
            train(m, ld_train, fl_rate = 1.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL20AUX03'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL50AUX03'
            train(m, ld_train, fl_rate = 5.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_explore_FL():
    PATH = 'run_explore_FL.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev()
    # 25 * 1 * 5 * 3 = 375 (mins) = 6.25 (hours)
    times = 1
    epochs = 3
    for _ in range(times): # Baseline BCE loss
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL00'
            train_baseline(m, ld_train, fl_rate = 0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL05'
            train_baseline(m, ld_train, fl_rate = 0.5)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL10'
            train_baseline(m, ld_train, fl_rate = 1.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL20'
            train_baseline(m, ld_train, fl_rate = 2.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(epochs):
            key = f'E{i+1}FL50'
            train_baseline(m, ld_train, fl_rate = 5.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)


def run1(): # 16小时
    run_explore_FL() # 6.25 hours
    run_explore_AUX() # 5 hours
    run_explore_FL_AUX00() # 5 hours


def run2(): # 15小时
    run_explore_FL_AUX01() # 5 
    run_explore_FL_AUX02() # 5
    run_explore_FL_AUX03() # 5 


def run_comparison():
    PATH = 'comparison.txt'
    times = 5
    ld_train = read_ld_train()
    ld_test = read_ld_test() # NOTE: 必须是test
    # 5 * (2 + 2 + 2 + 1) * 25 = 875(min) = 14.58(hour)
    for _ in range(times):
        m = Sector_2022()
        for i in range(2):
            key = f'E{i+1}FL'
            train_baseline(m, ld_train, fl_rate = 1.0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(2):
            key = f'E{i+1}STANDARD'
            train_baseline(m, ld_train, fl_rate = 0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(1):
            key = f'E{i+1}AUX_FL'
            train(m, ld_train, fl_rate = 2.0, aux_rate = 0.3)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(times):
        m = Sector_2022()
        for i in range(2):
            key = f'E{i+1}AUX'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)

def run_comparison_plus():
    PATH = 'comparison_plus.txt'
    times = 1
    ld_train = read_ld_train()
    ld_test = read_ld_test() # NOTE: 必须是test
    # 5 * (2 + 2 + 2 + 1) * 25 = 875(min) = 14.58(hour)
    for _ in range(times):
        m = Sector_2022()
        for i in range(2):
            key = f'E{i+1}STANDARD'
            train_baseline(m, ld_train, fl_rate = 0)
            dic[key].append(test_chain_baseline(m, ld_test))
        save_dic(PATH)

def run_comparison_plus2():
    PATH = 'comparison_plus.txt'
    times = 1
    ld_train = read_ld_train()
    ld_test = read_ld_test() # NOTE: 必须是test
    for _ in range(times):
        m = Sector_2022()
        for i in range(2):
            key = f'E{i+1}AUX'
            train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
            dic[key].append(test_chain(m, ld_test))
        save_dic(PATH)


