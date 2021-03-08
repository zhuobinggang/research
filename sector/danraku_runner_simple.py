import utils as U
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import time

# testld = data.Loader(data.Test_DS(), 1)
# ld = data.Loader(data.Train_DS(), 8)

def plot_loss(epoch, train_loss, test_loss):
  xs = [x+1 for x in list(range(epoch))]
  plt.clf()
  plt.plot(xs, train_loss, label= 'train loss')
  plt.plot(xs, test_loss, label= 'test loss')
  plt.legend()
  plt.savefig('result.png')
  

def plot_prec_rec_f1(results, loss, epoch= 20, path = 'result.png'):
  precs = []
  recs = []
  f1s = []
  baccs = []
  for prec, rec, f1, bacc in results:
    precs.append(prec)
    recs.append(rec)
    baccs.append(bacc)
    f1s.append(f1)
  # return precs, recs, f1s
  xs = [x+1 for x in list(range(epoch))]
  plt.plot(xs, precs, label= 'precs')
  plt.plot(xs, recs, label= 'recs')
  plt.plot(xs, f1s, label= 'f1s')
  plt.plot(xs, baccs, label= 'balanced acc')


def plot_prec_rec_f1_loss(results, loss, epoch= 20, path = 'result.png'):
  plt.clf()
  plot_prec_rec_f1(results, loss, epoch, path)
  xs = [x+1 for x in list(range(epoch))]
  plt.plot(xs, loss, label= 'loss')
  plt.legend()
  plt.savefig(path)


def logout_info(text):
  print(text)
  U.logging.info(text)

def run(m, ld, testld, devld = None, epoch = 2, log_path = 'wiki.log', batch=100):
  # Initialize
  U.init_logger(log_path)
  ld.start = 0 
  ld.batch_size = batch
  testld.start = 0
  testld.batch_size = batch
  devld.start = 0
  devld.batch_size = batch
  # Get result
  results = []
  losss = []
  for i in range(epoch):
    loss = U.train_by_data_loader_danraku_origin(m, ld, 1, U.logging.debug)[0]
    losss.append(loss)
    # Get test result
    result = {'test': get_test_result(m, testld), 'dev': get_test_result(m, devld)}
    results.append(result)
    logout_info(f'epoch = {i+1}, loss = {loss}, result = {result}')
  # plot_prec_rec_f1_loss(results, losss, epoch, path)
  # print('save to result.png')
  return m, results, losss


def train_simple(m, ld, epoch, log_path = 'wiki.log'):
  U.init_logger(log_path)
  ld.start = 0 
  losses = [U.train_by_data_loader_danraku_origin(m, ld, 1, U.logging.debug)[0] for _ in range(epoch)]
  return losses

def get_test_result(m, testld):
  testld.start = 0
  dic = {}
  if testld is None:
    dic['prec'] = -1
    dic['rec'] = -1
    dic['f1'] = -1
    dic['bacc'] = -1
  else: 
    outputs, targets = U.get_test_results_batch(m, testld, U.logging.debug)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    dic['prec'] = prec
    dic['rec'] = rec
    dic['f1'] = f1
    dic['bacc'] = bacc
  return dic


def get_test_result_segbot(m, testds):
  dic = {}
  if testds is None:
    dic['prec'] = -1
    dic['rec'] = -1
    dic['f1'] = -1
    dic['bacc'] = -1
  else: 
    outputs, targets = get_test_results_segbot_v2(m, testds, U.logging.debug)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    dic['prec'] = prec
    dic['rec'] = rec
    dic['f1'] = f1
    dic['bacc'] = bacc
  return dic

def get_test_results_segbot(m, ds, logger = print):
  targets = []
  results = []
  time_start = time.time()
  start = 0
  while start != -1 and start < len(ds.datas):
    inpts, labels = ds[start]
    ground_truth = labels.tolist().index(1)
    predict = m.dry_run(inpts, labels)
    targets += [1]
    results += ([0] if predict != ground_truth else [1])
    if predict == 0:
      start += 1 
    elif predict >= ds.ss_len:
      start += (predict - 1) # 防止错过换段
      print('防止错过换段')
    else:
      start += predict
    logger(f'Testing: {start}/{len(ds.datas)}')
  time_end = time.time()
  logger(f'Tested! Time cost: {time_end - time_start} seconds')
  return results, targets

def get_targets_by_ds(ds):
  targets = []
  for s in ds.datas:
    targets.append(1 if ds.is_begining(s) else 0)
  return targets

# Tested 基本没问题
def get_test_results_segbot_v2(m, ds, logger = print):
  targets = get_targets_by_ds(ds)
  results = np.zeros(len(targets), dtype=np.int8).tolist()
  time_start = time.time()
  start = 0
  while start != -1 and start < len(ds.datas):
    inpts, labels = ds[start]
    # ground_truth = labels.tolist().index(1)
    predict = m.dry_run(inpts, labels)
    # 根据predict填充分割点
    if predict == labels.shape[0] - 1: # 预测没有分割点
      pass
    else:
      results[start + predict] = 1
    # 移动指针
    if predict == 0:
      start += 1 
    elif predict >= ds.ss_len: # (预测: 没有分割点)
      start += (predict - 1) # 防止错过换段
      print('防止错过换段')
    else:
      start += predict
    logger(f'Testing: {start}/{len(ds.datas)}')
  time_end = time.time()
  logger(f'Tested! Time cost: {time_end - time_start} seconds')
  return results, targets
