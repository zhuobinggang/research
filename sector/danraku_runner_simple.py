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


def run(m, ld, testld, epoch = 2, path = 'result.png', log_path = 'wiki.log', batch=100):
  U.init_logger(log_path)
  ld.start = 0 
  ld.batch_size = batch
  testld.start = 0
  testld.batch_size = 1
  results = []
  losss = []
  for i in range(epoch):
    loss = U.train_by_data_loader_danraku_origin(m, ld, 1, U.logging.debug)[0]
    losss.append(loss)
    outputs, targets = U.get_test_results_batch(m, testld)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    results.append((prec, rec, f1, bacc))
    print(f'epoch = {i}, loss = {loss}, {prec}, {rec}, {f1}')
  plot_prec_rec_f1_loss(results, losss, epoch, path)
  print('save to result.png')
  return m, results, losss
