import result_092721 as R1
import result_101821 as R2
import numpy as np

def get_results_raw():
    xs, [ys_dev, ys_test], legends = R1.get_results()
    ys_dev = ys_dev.tolist()
    ys_test = ys_test.tolist()
    dic1 = {}
    for x, y_dev, y_test in zip(xs, ys_dev, ys_test):
        dic1[x] = {}
        dic1[x]['y_dev'] = y_dev
        dic1[x]['y_test'] = y_test
    xs, [ys_dev, ys_test], legends = R2.get_results()
    ys_dev = ys_dev.tolist()
    ys_test = ys_test.tolist()
    dic2 = {}
    for x, y_dev, y_test in zip(xs, ys_dev, ys_test):
        dic2[x] = {}
        dic2[x]['y_dev'] = y_dev
        dic2[x]['y_test'] = y_test
    # 缝合
    # xs = [] + xs1 + xs2
    # ys_dev = np.concatenate([ys_dev1, ys_dev2])
    # ys_test = np.concatenate([ys_test1, ys_test2])
    # legends = [] + legends1 + legends2
    # return xs, [ys_dev, ys_test], legends
    return {**dic1, **dic2}


def get_results():
    dic = get_results_raw()
    # xs = ['r00+fl10', 'r00+fl20', 'r00+fl05', 'r01+fl10','r01+fl20','r01+fl05','r02+fl10','r02+fl20', 'r02+fl05', 'stand', 'fl10', 'fl20', 'fl05', 'r00', 'r01', 'r02', 'fl30', 'fl40', 'fl50', 'r00+fl30','r00+fl40','r00+fl50','r01+fl30','r01+fl40', 'r01+fl50', 'r02+fl30', 'r02+fl40', 'r02+fl50']
    xs = ['stand']
    for fl in ['05', '10', '20', '30', '40', '50']:
        xs.append(f'fl{fl}')
    for r in ['00', '01', '02']:
        xs.append(f'r{r}')
    for r in ['00', '01', '02']:
        for fl in ['05', '10', '20', '30', '40', '50']:
            xs.append(f'r{r}+fl{fl}')
    ys_dev = []
    for key in xs:
        ys_dev.append(dic[key]['y_dev'])
    ys_test = []
    for key in xs:
        ys_test.append(dic[key]['y_test'])
    legends = ['y_dev', 'y_test']
    return xs, [ys_dev, ys_test], legends



