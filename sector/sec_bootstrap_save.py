from sec_paragraph import *
import random

init_G_Symmetry_Mainichi(half=2, batch=1, mini=False)
tld = G['testld']
# tld = G['devld']

g_save = {
        'tld': tld, 
        'targets': None, 
        'outputs_aux_fl': [], 
        'outputs_aux': [], 
        'outputs_fl':[],
        'outputs_stand': []
}

def cal_results(start_index = 0, cnt = 10):
    global g_save
    # aux + fl
    print('aux + fl')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/r01_fl20_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_aux_fl'].append(outputs)
    # stand
    print('stand')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/stand_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_stand'].append(outputs)
    # fl
    print('fl')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/fl20_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_fl'].append(outputs)
    # aux
    print('aux')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/r01_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_aux'].append(outputs)
    # save
    save_g(start_index, cnt)

def save_g(start_index, cnt):
    global g_save
    f = open(f'boot_results_by_model.2v2.txt', 'w')
    f.write(str(g_save))

