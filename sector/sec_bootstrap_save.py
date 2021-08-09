from sec_paragraph import *
import random

init_G_Symmetry_Mainichi(half=2, batch=1, mini=False)
tld = G['testld']

g_save = {
        'targets': None, 
        'outputs_stand': [], 
        'outputs_my': [], 
        'outputs_FL':[] 
}

def cal_results(start_index = 0):
    global g_save
    # stand
    for i in range(5):
        idx = start_index + i
        m = t.load(f'save/stand_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_stand'].append(outputs)
    # my
    for i in range(5):
        idx = start_index + i
        m = t.load(f'save/my_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_my'].append(outputs)
    # fl
    for i in range(5):
        idx = start_index + i
        m = t.load(f'save/fl_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_FL'].append(outputs)
    # save
    save_g(start_index)

def save_g(start_index):
    global g_save
    f = open(f'bootstrap_results_start_from_{start_index}.txt', 'w')
    f.write(str(g_save))



