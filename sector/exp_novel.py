from sector import *

dic = {
  'E1FL0AUX00': [],        
  'E2FL0AUX00': [],       
  'E3FL0AUX00': [],       
  'E1FL0AUX01': [],       
  'E2FL0AUX01': [],     
  'E3FL0AUX01': [], 
  'E1FL0AUX02': [],       
  'E2FL0AUX02': [],      
  'E3FL0AUX02': [], 
  'E1FL0AUX03': [],       
  'E2FL0AUX03': [],     
  'E3FL0AUX03': [],

  'E1FL05AUX00': [],        
  'E2FL05AUX00': [],       
  'E3FL05AUX00': [],       
  'E1FL10AUX00': [],       
  'E2FL10AUX00': [],     
  'E3FL10AUX00': [], 
  'E1FL20AUX00': [],       
  'E2FL20AUX00': [],      
  'E3FL20AUX00': [], 
  'E1FL50AUX00': [],       
  'E2FL50AUX00': [],     
  'E3FL50AUX00': [],

  'E1FL00': [],
  'E2FL00': [],
  'E3FL00': [],
  'E4FL00': [],
  'E5FL00': [],
  'E1FL05': [],
  'E2FL05': [],
  'E3FL05': [],
  'E1FL10': [],
  'E2FL10': [],
  'E3FL10': [],
  'E1FL20': [],
  'E2FL20': [],
  'E3FL20': [],
  'E1FL50': [],
  'E2FL50': [],
  'E3FL50': [],
}

def save_dic(name = 'exp_novel.txt'):
    f = open(name, 'w')
    f.write(str(dic))
    f.close()

def run_explore_AUX():
    PATH = 'run_explore_AUX.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 3 * 4 = 900 (mins) = 15 (hours)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0)
        dic['E1FL0AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0)
        dic['E2FL0AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0)
        dic['E3FL0AUX00'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
        dic['E1FL0AUX01'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
        dic['E2FL0AUX01'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
        dic['E3FL0AUX01'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        dic['E1FL0AUX02'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        dic['E2FL0AUX02'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        dic['E3FL0AUX02'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
        dic['E1FL0AUX03'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
        dic['E2FL0AUX03'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
        dic['E3FL0AUX03'].append(test_chain(m, ld_test))
        save_dic(PATH)


def run_explore2():
    ld_train = read_ld_train()
    ld_test = read_ld_dev() # grid search
    # 25 * 3 * 3 * 4 = 900 (mins) = 15 (hours)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0.5, aux_rate = 0)
        dic['E1FL05AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0.5, aux_rate = 0)
        dic['E2FL05AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0.5, aux_rate = 0)
        dic['E3FL05AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 1.0, aux_rate = 0)
        dic['E1FL10AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 1.0, aux_rate = 0)
        dic['E2FL10AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 1.0, aux_rate = 0)
        dic['E3FL10AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 2.0, aux_rate = 0)
        dic['E1FL20AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 2.0, aux_rate = 0)
        dic['E2FL20AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 2.0, aux_rate = 0)
        dic['E3FL20AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 5.0, aux_rate = 0)
        dic['E1FL50AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 5.0, aux_rate = 0)
        dic['E2FL50AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 5.0, aux_rate = 0)
        dic['E3FL50AUX00'].append(test_chain(m, ld_test))
        save_dic()


def run_explore_FL():
    PATH = 'run_explore_FL.txt'
    ld_train = read_ld_train()
    ld_test = read_ld_dev()
    # 25 * 3 * 3 * 5 = 1125 (mins) = 18.75 (hours)
    for _ in range(3): # Baseline BCE loss
        m = Sector_2022() 
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0)
        dic['E1FL00'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0)
        dic['E2FL00'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0)
        dic['E3FL00'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0)
        dic['E4FL00'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0)
        dic['E5FL00'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0.5)
        dic['E1FL05'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0.5)
        dic['E2FL05'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 0.5)
        dic['E3FL05'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 1.0)
        dic['E1FL10'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 1.0)
        dic['E2FL10'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 1.0)
        dic['E3FL10'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 2.0)
        dic['E1FL20'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 2.0)
        dic['E2FL20'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 2.0)
        dic['E3FL20'].append(test_chain(m, ld_test))
        save_dic(PATH)
    for _ in range(3):
        m = Sector_2022()
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 5.0)
        dic['E1FL50'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 5.0)
        dic['E2FL50'].append(test_chain(m, ld_test))
        train_baseline(m, ld_train, epoch = 1, batch = 16, fl_rate = 5.0)
        dic['E3FL50'].append(test_chain(m, ld_test))
        save_dic(PATH)



def run_explore4():
    ld_train = read_ld_train()
    ld_test = read_ld_dev()
    # 25 * 3 * 3 * 4 = 900 (mins) = 15 (hours)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0.5, aux_rate = 0.1)
        dic['E1FL05AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0.5, aux_rate = 0.1)
        dic['E2FL05AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0.5, aux_rate = 0.1)
        dic['E3FL05AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 1.0, aux_rate = 0.1)
        dic['E1FL10AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 1.0, aux_rate = 0.1)
        dic['E2FL10AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 1.0, aux_rate = 0.1)
        dic['E3FL10AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 2.0, aux_rate = 0.1)
        dic['E1FL20AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 2.0, aux_rate = 0.1)
        dic['E2FL20AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 2.0, aux_rate = 0.1)
        dic['E3FL20AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
        dic['E1FL50AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
        dic['E2FL50AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 5.0, aux_rate = 0.1)
        dic['E3FL50AUX00'].append(test_chain(m, ld_test))
        save_dic()
