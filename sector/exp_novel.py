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
}

def save_dic():
    f = open('exp_novel.txt', 'w')
    f.write(str(dic))
    f.close()

def run_explore1():
    ld_train = read_ld_train()
    ld_test = read_ld_test()
    # 25 * 3 * 3 * 4 = 900 (mins) = 15 (hours)
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0)
        dic['E1FL0AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0)
        dic['E2FL0AUX00'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0)
        dic['E3FL0AUX00'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
        dic['E1FL0AUX01'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
        dic['E2FL0AUX01'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.1)
        dic['E3FL0AUX01'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        dic['E1FL0AUX02'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        dic['E2FL0AUX02'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.2)
        dic['E3FL0AUX02'].append(test_chain(m, ld_test))
        save_dic()
    for _ in range(3):
        m = Sector_2022()
        train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
        dic['E1FL0AUX03'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
        dic['E2FL0AUX03'].append(test_chain(m, ld_test))
        train(m, ld_train, fl_rate = 0, aux_rate = 0.3)
        dic['E3FL0AUX03'].append(test_chain(m, ld_test))
        save_dic()


def run_explore2():
    ld_train = read_ld_train()
    ld_test = read_ld_test()
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


