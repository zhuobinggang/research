from sec_paragraph import *
import random

init_G_Symmetry_Mainichi(half=2, batch=1, mini=False)
tld = G['testld']

TLD = 'TLD'
TARGETS = 'targets'
OUTPUTS_AUX_FL = 'outputs_aux_fl'
OUTPUTS_AUX = 'outputs_aux'
OUTPUTS_FL = 'outputs_fl'
OUTPUTS_STAND = 'outputs_stand'

g_save = {
        TLD: tld, 
        TARGETS: None, 
        OUTPUTS_AUX_FL: [], 
        OUTPUTS_AUX: [], 
        OUTPUTS_FL:[],
        OUTPUTS_STAND: []
}

def cal_results(start_index = 0, cnt = 10):
    global g_save
    # aux + fl
    print('aux + fl')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/r01_fl20_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save[TARGETS] = targets
        g_save[OUTPUTS_AUX_FL].append(outputs)
    # stand
    print('stand')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/stand_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save[TARGETS] = targets
        g_save[OUTPUTS_STAND].append(outputs)
    # fl
    print('fl')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/fl20_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save[TARGETS] = targets
        g_save[OUTPUTS_FL].append(outputs)
    # aux
    print('aux')
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/r01_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save[TARGETS] = targets
        g_save[OUTPUTS_AUX].append(outputs)
    # save
    save_g(start_index, cnt)

def save_g(start_index, cnt):
    global g_save
    f = open(f'boot_results_by_model.2v2.txt', 'w')
    f.write(str(g_save))

