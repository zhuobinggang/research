import data_jap as data
import danraku2 as model
import utils as U
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


testld = data.Loader(data.Test_DS(), 1)
ld = data.Loader(data.Train_DS(), 8)
mini_testld = data.Loader(data.Test_DS_Mini(), 1)
mini_ld = data.Loader(data.Train_DS_Mini(), 8)

def run(m, epoch = 20, path = 'result.png'):
  U.init_logger('danraku.log')
  results = []
  losss = []
  for i in range(epoch):
    loss = U.train_by_data_loader_danraku(m, ld, 1, U.logging.debug)[0]
    losss.append(loss)
    outputs, targets = U.get_test_results(m, testld)
    prec, rec, f1 = U.cal_prec_rec_f1(outputs, targets)
    results.append((prec, rec, f1))
  plot_prec_rec_f1(results, epoch, path)
  print('save to result.png')
  return m, results, losss

def run_test(m, epoch = 1, path = 'result.png'):
  U.init_logger('danraku.log')
  results = []
  losss = []
  for i in range(epoch):
    loss = U.train_by_data_loader_danraku(m, mini_ld, 1, U.logging.debug)[0]
    losss.append(loss)
    outputs, targets = U.get_test_results(m, mini_testld)
    prec, rec, f1 = U.cal_prec_rec_f1(outputs, targets)
    results.append((prec, rec, f1))
  plot_prec_rec_f1(results, epoch, path)
  print('save to result.png')
  return m, results, losss


def run_plot_loss(epoch = 20):
  m = model.Model_Con()
  U.init_logger('danraku.log')
  train_loss = []
  test_loss = []
  for i in range(epoch):
    train_loss.append(U.train_by_data_loader_danraku(m, ld, 1, U.logging.debug)[0])
    test_loss.append(U.get_avg_loss(m, testld))
  plot_loss(epoch, train_loss, test_loss)
  print('save to loss_epoch.png')
  return m, train_loss, test_loss

def plot_loss(epoch, train_loss, test_loss):
  xs = [x+1 for x in list(range(epoch))]
  plt.clf()
  plt.plot(xs, train_loss, label= 'train loss')
  plt.plot(xs, test_loss, label= 'test loss')
  plt.legend()
  plt.savefig('result.png')
  

def plot_prec_rec_f1(results, epoch= 20, path = 'result.png'):
  precs = []
  recs = []
  f1s = []
  for prec, rec, f1 in results:
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
  # return precs, recs, f1s
  xs = [x+1 for x in list(range(epoch))]
  plt.plot(xs, precs, label= 'precs')
  plt.plot(xs, recs, label= 'recs')
  plt.plot(xs, f1s, label= 'f1s')
  plt.legend()
  plt.savefig(path)

