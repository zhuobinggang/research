# 根据4*10个模型*10个数据集的结果获取4*10个数据集的平均f值
# from messup import res
# from mess import res
from mainichi_paragraph import *
import numpy as np

# shape: (4, 40, 10, ?) 
# out: (4, 10)


def cal_prec_rec_f1_v2(results, targets):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  for guess, target in zip(results, targets):
    if guess == 1:
      if target == 1:
        TP += 1
      elif target == 0:
        FP += 1
    elif guess == 0:
      if target == 1:
        FN += 1
      elif target == 0:
        TN += 1
  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  rec = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
  balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
  balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
  return prec, rec, f1, balanced_acc

def cal_f1(outs, tars):
    prec, rec, f1, balanced_acc = cal_prec_rec_f1_v2(outs, tars)
    return f1

def get_targets():
    return read_additional_test_dataset_targets()

def cal(res):
    targets = get_targets()
    methodn = []
    for method in res: # (40, 10, ?)
        testm_avg_f1 = []
        for test_number in range(10):
            f1s = []
            for model in method: # (10, ?)
                outputs = model[test_number] # (?)
                f1 = cal_f1(outputs, targets[test_number])
                f1s.append(f1)
            avg_f1 = np.average(f1s)
            testm_avg_f1.append(avg_f1)
        methodn.append(testm_avg_f1)
    return np.array(methodn).transpose()


