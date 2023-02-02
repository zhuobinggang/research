from res_4method_10_m_10_testset import res # 新闻数据集，论文表里面的结果
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import mainichi_paragraph as custom_data
from get_f_by_188_chapters import dic

def get_ground_truth():
    y_truess = []
    for i in range(10):
        y_trues = []
        ds = custom_data.load_customized_loader(file_name = f'test{i}', half = 2, batch = 1, shuffle = False)
        for item in ds:
            ss, ls, pos = item[0]
            if pos != 0:
                y_trues.append(ls[pos])
        y_truess.append(y_trues)
    return y_truess

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



# res_4method_10_m_10_testset
def cal_results():
    ground_trues = get_ground_truth()
    results = []
    for method in res:
        models = []
        for model in method:
            testsets = []
            for testset, y_true in zip(model, ground_trues):
                prec_rec_f_bacc = cal_prec_rec_f1_v2(testset, y_true)
                testsets.append(prec_rec_f_bacc)
            models.append(testsets)
        results.append(models)
    return results


# 4method 10m 10testset prec_rec_f_bacc
# 4method: auxfl, fl, stand, aux
def run():
    results = np.array(cal_results())
    print(results.shape) # (4, 10, 10, 4)
    results = results.mean(1) # (4, 10, 4)
    # all one
    ground_trues = get_ground_truth()
    all_one = []
    for trues in ground_trues:
        all_one.append(cal_prec_rec_f1_v2([1] * len(trues), trues))
    all_one = np.array(all_one)




################### 小说


def dd():
    # from get_f_by_188_chapters import dic
    auxfl = dic['AUX_FL188'] # (10, 188, (prec, rec, f))
    aux = dic['AUX188']
    fl = dic['FL188']
    stand = dic['STAND188']
    fl.mean(0).mean(0) # array([0.86234496, 0.81707165, 0.82691926])

