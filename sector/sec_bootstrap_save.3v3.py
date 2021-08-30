from sec_paragraph import *
import random

init_G_Symmetry_Mainichi(half=3, batch=2, mini=False)
tld = G['testld']

g_save = {
        'tld': tld, 
        'targets': None, 
        'outputs':[] 
}

def cal_results(start_index = 0, cnt = 10):
    global g_save
    for i in range(5):
        idx = i
        m = t.load(f'save/my_3v3_{idx}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save['targets'] = targets
        g_save['outputs'].append(outputs)
    # save
    save_g(start_index, cnt)

def save_g(start_index, cnt):
    global g_save
    f = open(f'g_save.3v3.txt', 'w')
    f.write(str(g_save)) 
