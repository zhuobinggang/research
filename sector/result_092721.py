import numpy as np

from mess import *

def get_mat(results, key = 'f1'):
    mat = []
    for item in results:
        mat.append((item[key], item['test_result'][key]))
    mat = np.array(mat)
    return mat
