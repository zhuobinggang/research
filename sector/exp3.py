# 重新跑了一次grid_search

from mess_pc import res_pc
from mess_pt import res_pt
import numpy as np

def dic_to_tuple(dic):
    return (dic['f_dev'], dic['f_test'])

# return: (10, 3, 2)
# array([[0.66048746, 0.66911682],
#        [0.67773997, 0.68427391],
#        [0.67268398, 0.6818272 ]])
# 分析：最好的是(epoch: 2)，dev和test没有区别
# 不需要重新训练stand
# exp0: array([0.66480214, *0.68100694, 0.67725559]) #NOTE: trained
# exp1: array([0.66093464, 0.67162706, *0.68659469])
def transform_bce(res = res_pc):
    bce = res[0].copy()
    for m in bce:
        for i, e in enumerate(m):
            m[i] = dic_to_tuple(e)
    return np.array(bce)

# [0.1, 0.2, 0.3]
# array([[0.65250394, 0.67939609, 0.68450577],
#        [0.6642753 , 0.68338524, 0.68321264],
#        [0.67711231, *0.68990987, 0.67820931],
#        [0.67801763, 0.68657151, 0.68233489]])
# 分析：最好的是(aux: 0.2, epoch: 2), 需要重新训练model
# return: (4, 10, 3, 2)
def transform_aux(res = res_pc):
    aux = res[1].copy() # shape: (3, 10, 3)
    for aux_rate in aux:
        for i, number in enumerate(aux_rate):
            aux_rate[i] = [(epoch['f_dev'], epoch['f_test']) for epoch in number]
    return np.array(aux)


# fls = [0.5, 1.0, 2.0, 5.0]
# return: (4, 10, 3, 2)
# array([[[0.64680301, 0.65583285],
#         [0.67631652, 0.68752971],
#         [0.6757291 , 0.68871174]],
# 
#        [[0.65630211, 0.66213435],
#         [0.6744992 , 0.68448335],
#         [0.67122137, 0.68548532]],
# 
#        [[0.67176207, 0.67939854],
#         [0.67579604, 0.68811805],
#         [0.68068657, 0.69140937]],
# 
#        [[0.64616078, 0.65565814],
#         [0.67633121, 0.68438043],
#         [0.67814608, 0.68622593]]])
#
# 如果只按照dev数据集来看的话，fls = 2.0，epoch = 3的时候最好
# 如果只按照test数据集来看的话，同上
# 如果用两个数据集平均来看的话，fls = 2.0, epoch = 3的时候最好
# 怎么解释之前的结果？没有差很多，不过e3性能最好，得小心了。。。需要重新训练
# array([[0.65131793, 0.68192311, 0.68222042],
#        [0.65921823, 0.67949128, 0.67835334],
#        [0.6755803 , 0.68195704, *0.68604797],
#        [0.65090946, 0.68035582, 0.682186  ]])
def transform_fl(res = res_pc):
    fl = res[2].copy() # shape: (4, 10, 3)
    for method in fl:
        for m in method:
            for i, epoch in enumerate(m):
                m[i] = (epoch['f_dev'], epoch['f_test'])
    return np.array(fl)

def transform_fl_free(fl):
    fl = fl.copy()
    for method in fl:
        for m in method:
            for i, epoch in enumerate(m):
                m[i] = (epoch['f_dev'], epoch['f_test'])
    return np.array(fl)
                

# auxs = [0.0, 0.1, 0.2], 
# fls = [0.5, 1.0, 2.0, 5.0], 
# return (4, 4, 10, 3, 2)
# a00 最好是5.0 + 3
# array([[0.64530309, 0.67363298, 0.68275966],
#        [0.66997731, 0.68167708, 0.68920777],
#        [0.62353709, 0.68033193, 0.68567287],
#        [0.65163644, 0.68873465, *0.6920493 ]])
# a01 最好是5.0 + 2
# array([[0.66550531, 0.68102308, 0.68377398],
#        [0.67636772, 0.68665621, 0.68604364],
#        [0.66511946, 0.69666519, 0.68944727],
#        [0.68058342, *0.69765496, 0.6954732 ]])
# a02 best 5.0 + 2
# array([[0.66612304, 0.69190034, 0.68156865],
#        [0.67572537, 0.68955983, 0.68111424],
#        [0.68568132, 0.69242442, 0.67697666],
#        [0.69341916, *0.69362768, 0.68691613]])
# a03 best 1.0 + 2
# array([[0.68403235, 0.67611857, 0.68772222],
#        [0.67613924, *0.69291704, 0.68345438],
#        [0.67364332, 0.69268254, 0.68508181],
#        [0.69024683, 0.68994638, 0.68335526]])
# 而总体来说最好是a01 + 5.0 + 2，意外的不需要重新训练
def transform_fl_aux(res = res_pt):
    res = res[0].copy() 
    for aux in res:
        for fl in aux:
            for m in fl:
                for i, e in enumerate(m):
                    m[i] = dic_to_tuple(e)
    return np.array(res)


# Want: 4 * 10 * 10
def test():
    pass

