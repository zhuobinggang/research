import boot_results_by_model_2v2
g_save = boot_results_by_model_2v2.res

TLD = 'TLD'
TARGETS = 'targets'
OUTPUTS_AUX_FL = 'outputs_aux_fl'
OUTPUTS_R01_FL50 = 'outputs_r01_fl50'
OUTPUTS_AUX = 'outputs_aux'
OUTPUTS_FL = 'outputs_fl'
OUTPUTS_STAND = 'outputs_stand'

tld = g_save[TLD]

def cal_and_save_for_r01fl50_10_times():
    start_index = 0
    for i in range(10):
        idx = start_index + i
        m = t.load(f'save/r01_fl50_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save[TARGETS] = targets
        g_save[OUTPUTS_R01_FL50].append(outputs)
    f = open(f'boot_results_by_model.2v2.txt', 'w')
    f.write(str(g_save))
    f.close()


def cal_and_resave_40_times():
    print('r01_fl50')
    for i in range(10):
        idx = start_index + i
        m = t.load(f'save/r01_fl50_{i + start_index}.tch')
        outputs, targets = get_test_result(m, tld)
        g_save[TARGETS] = targets
        g_save[OUTPUTS_R01_FL50].append(outputs)
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
    f.close()


