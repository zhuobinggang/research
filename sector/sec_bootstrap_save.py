from sec_paragraph import *
import random

init_G_Symmetry_Mainichi(half=2, batch=1, mini=False)
# tld = G['testld']
tld = G['devld']

g_save = {
        'tld': tld, 
        'targets': None, 
        'outputs_stand': [], 
        'outputs_my': [], 
        'outputs_FL':[],
        'outputs_myr0': []
}

def cal_results(start_index = 0, cnt = 10):
    global g_save
    # stand
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/stand_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_stand'].append(outputs)
    # my
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/my_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_my'].append(outputs)
    # my r00
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/my_r00_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_myr0'].append(outputs)
    # fl
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/fl_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs_FL'].append(outputs)
    # save
    save_g(start_index, cnt)

def save_g(start_index, cnt):
    global g_save
    f = open(f'g_save.2v2.txt', 'w')
    f.write(str(g_save))


def cal_results_new(start_index = 0, cnt = 10):
    global g_save
    # stand
    for i in range(cnt):
        idx = start_index + i
        m = t.load(f'save/my_r00_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs'].append(outputs)
