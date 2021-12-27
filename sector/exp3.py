from mess_pc import res_pc
import numpy as np

def dic_to_tuple(dic):
    return (dic['f_dev'], dic['f_test'])

# return: (10, 3, 2)
# array([[0.66048746, 0.66911682],
#        [0.67773997, 0.68427391],
#        [0.67268398, 0.6818272 ]])
# 分析：最好的是(epoch: 2)，dev和test没有区别
# 不需要重新训练stand
# array([0.66480214, *0.68100694, 0.67725559])
def transform_bce():
    bce = res_pc[0].copy()
    for m in bce:
        for i, e in enumerate(m):
            m[i] = dic_to_tuple(e)
    return np.array(bce)

# [0.1, 0.2, 0.3]
# array([[0.6642753 , 0.68338524, 0.68321264],
#        [0.67711231, *0.68990987, 0.67820931],
#        [0.67801763, 0.68657151, 0.68233489]])
# 分析：最好的是(aux: 0.2, epoch: 2), 需要重新训练model
# return: (3, 10, 3, 2)
def transform_aux():
    aux = res_pc[1].copy() # shape: (3, 10, 3)
    for aux_rate in aux:
        for i, number in enumerate(aux_rate):
            aux_rate[i] = [(epoch['f_dev'], epoch['f_test']) for epoch in number]
    return aux


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
# 怎么解释之前的结果？没有差很多，不过e3性能最好，得小心了。。。
# array([[0.65131793, 0.68192311, 0.68222042],
#        [0.65921823, 0.67949128, 0.67835334],
#        [0.6755803 , 0.68195704, *0.68604797],
#        [0.65090946, 0.68035582, 0.682186  ]])
def transform_fl():
    fl = res_pc[2].copy() # shape: (4, 10, 3)
    for method in fl:
        for m in method:
            for i, epoch in enumerate(m):
                m[i] = (epoch['f_dev'], epoch['f_test'])
    return np.array(fl)
                

