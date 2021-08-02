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

def cal_results():
    global g_save
    # stand
    for i in range(5):
        m = t.load(f'save/stand_{i}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_stand'].append(outputs)
    # my
    for i in range(5):
        m = t.load(f'save/my_{i}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_my'].append(outputs)
    # fl
    for i in range(5):
        m = t.load(f'save/fl_{i}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_FL'].append(outputs)


