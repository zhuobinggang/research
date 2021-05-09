import logging
import time
import numpy as np
import requests
import torch
t = torch

def init_logger(path):
  logging.basicConfig(
    filename=path,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

def request_my_logger(dic, desc = 'No describe', url = None):
  if url is None:
    pass
    # url = default_url
  else:
    try:
      dic['desc'] = desc
      requests.post(url, json=dic)
    except:
      print('Something went wrong in request_my_logger()')

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

def get_test_result(m, loader):
  logger = logging.debug
  loader.start = 0
  start = time.time()
  length = len(loader.ds.datas)
  outputs = []
  targets = []
  for mass in loader:
    logger(f'TESTING: {loader.start}/{length}')
    out, labels = m.dry_run(mass)
    outputs += out.tolist()
    targets += labels.tolist()
  end = time.time()
  logger(f'TESTED! length={length} Time cost: {end - start} seconds')
  return outputs, targets

def get_test_result_dic(m, testld):
  testld.start = 0
  dic = {}
  if testld is None:
    dic['prec'] = -1
    dic['rec'] = -1
    dic['f1'] = -1
    dic['bacc'] = -1
  else: 
    outputs, targets = get_test_result(m, testld)
    prec, rec, f1, bacc = cal_prec_rec_f1_v2(outputs, targets)
    dic['prec'] = prec
    dic['rec'] = rec
    dic['f1'] = f1
    dic['bacc'] = bacc
  return dic

def cal_valid_loss(m, loader):
  logger = logging.debug
  loader.start = 0
  start = time.time()
  length = len(loader.ds.datas)
  loss = 0
  for mass in loader:
    logger(f'VALID: {loader.start}/{length}')
    losses = m.get_loss(mass)
    if len(losses) < 1:
      loss += 0
    else:
      loss += t.stack(losses).detach().sum().item()
  end = time.time()
  logger(f'VALIDED! length={length} Time cost: {end - start} seconds')
  return loss


def train_simple(m, loader, epoch):
  logger = logging.debug
  loss_per_epoch = []
  loader.start = 0
  start = time.time()
  length = len(loader.ds.datas)
  for e in range(epoch):
    loader.shuffle()
    logger(f'start epoch{e}')
    total_loss = 0
    for mass in loader:
      logger(f'{loader.start}/{length}')
      total_loss += m.train(mass)
    avg_loss = total_loss / length
    loss_per_epoch.append(total_loss)
  end = time.time()
  logger(f'Trained! Epochs: {epoch}, Batch size: {loader.batch_size}, dataset length: {length}, Time cost: {end - start} seconds')
  return loss_per_epoch

def get_datas_early_stop(m, ld, vld, tld, index, epochs, desc, dic_to_send = None, url = None):
  valid_losses = []
  train_losses = []
  tested = []
  results = []
  for i in range(epochs):
    train_losses += train_simple(m, ld, 1)
    valid_loss = cal_valid_loss(m, vld)
    valid_losses.append(valid_loss)
    dic_to_analyse = get_test_result_dic(m, tld)
    dic_to_analyse['index'] = i # Save index info
    dic_to_analyse['valid_loss'] = valid_loss # Save index info
    tested.append(dic_to_analyse)
  test_result = tested[np.argmin(valid_losses)]
  # test_result['valid_losses'] = valid_losses
  results.append(test_result) # 将valid loss最小对应的dic放进mess_list
  dic = test_result
  if dic_to_send is not None:
    dic = {**dic, **dic_to_send}
  else:
    pass
  request_my_logger(dic, desc, url)
  return results

