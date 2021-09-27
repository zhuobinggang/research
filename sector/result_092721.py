import numpy as np

from mess import *

def get_mat(results):
    mat = []
    for item in results:
        mat.append((item['f1'], item['test_result']['f1']))
    mat = np.array(mat)
    return mat
