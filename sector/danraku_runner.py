import data_jap as data
import danraku2 as model
danraku2 = model
import danraku3
import danraku4
import danraku5 # Juman++
import utils as U
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as t
import time


testld = data.Loader(data.Test_DS(), 1)
ld = data.Loader(data.Train_DS(), 8)
devld = data.Loader(data.Dev_DS(), 1)
ld2 = data.Loader(data.Train_DS(2), 8)
mini_testld = data.Loader(data.Test_DS_Mini(), 1)
mini_ld = data.Loader(data.Train_DS_Mini(), 8)

def run(m, epoch = 20, path = 'result.png', log_path = 'danraku.log', ld = ld, batch=8):
  U.init_logger(log_path)
  results = []
  losss = []
  recover_batch_size = ld.batch_size
  ld.batch_size = batch
  print(f'Loader window size: {ld.ds.half_window_size}')
  for i in range(epoch):
    loss = U.train_by_data_loader_danraku(m, ld, 1, U.logging.debug)[0]
    losss.append(loss)
    outputs, targets = U.get_test_results(m, testld)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    results.append((prec, rec, f1, bacc))
    print(f'epoch = {epoch}, loss = {loss}, {prec}, {rec}, {f1}')
  plot_prec_rec_f1_loss(results, losss, epoch, path)
  print('save to result.png')
  ld.batch_size = recover_batch_size
  return m, results, losss

def run_test(m, epoch = 1, path = 'result.png', log_path = 'danraku.log'):
  U.init_logger(log_path)
  results = []
  losss = []
  for i in range(epoch):
    loss = U.train_by_data_loader_danraku(m, mini_ld, 1, U.logging.debug)[0]
    losss.append(loss)
    outputs, targets = U.get_test_results(m, mini_testld)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    results.append((prec, rec, f1, bacc))
  plot_prec_rec_f1_loss(results, losss, epoch, path)
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

  


# 层级BERT，1 sentence per side
def run_cat_sentence():
  # m = t.load('save/bert_catsentence_epoch3.tch')
  m = danraku3.BERT_Cat_Sentence()
  _, results, losss = run(m, 4, 'bert_catsentence_epoch4_3vs1.png', 'danraku.log')
  t.save(m, 'save/bert_catsentence_1vs3.tch')
  print('run_cat_sentence from 1 -> 4 epoch over')
  return m, results, losss

def run_cross_seg():
  # m = t.load('save/bert_cross_seg_epoch3.tch')
  m = model.Model_Bert_Balanced_CE()
  _, results, losss = run(m, 4, 'bert_cross_seg_epoch4_3vs1.png', 'danraku.log')
  t.save(m, 'save/bert_cross_seg_1vs3.tch')
  print('run_cross_seg from 1 -> 4 epoch over')
  return m, results, losss

# 层级word2vec，1 sentence per side
def run_word2vec():
  m = t.load('save/wiki2vec.tch')
  _, results, losss = run(m, 5, 'wiki2vec_feb10_epoch10.png', 'word2vec.log')
  t.save(m, 'save/wiki2vec.tch')
  print('run_word2vec over')
  return m, results, losss

# 层级word2vec，1 sentence per side
def run_word2vec_2sentence_perside(m = danraku4.Model_Wiki2vec()):
  _, results, losss = run(m, 5, 'wiki2vec_feb7_sentence2_epoch8.png', 'word2vec.log', ld = ld2)
  t.save(m, 'save/wiki2vec_sentence2.tch')
  print('run_word2vec s2 over')
  return m, results, losss

def run_kuro(m = danraku5.BERT_Kuro()):
  _, results, losss = run(m, 3, 'save/kurobert_epoch3.png', 'kurobert.log', ld = ld)
  t.save(m, 'save/kurobert.tch')
  print('kurobert over')
  return m, results, losss

def run_kuro_catss(m = danraku5.Kuro_Catsentence()):
  _, results, losss = run(m, 3, 'save/kurobert_catss_epoch3.png', 'kurobert.log', ld = ld)
  t.save(m, 'save/kurobert_catss.tch')
  print('kurobert catss over')
  return m, results, losss

def run_feb10_night():
  # run_cross_seg()
  # run_cat_sentence()
  run_kuro()
  run_kuro_catss()

def test_test(m):
  start = time.time()
  outputs, targets = U.get_test_results(m, testld)
  end = time.time()
  print(f'Time cost: {end - start} seconds')
  return outputs, targets

def test_dev(m):
  start = time.time()
  outputs, targets = U.get_test_results(m, devld)
  end = time.time()
  print(f'Time cost: {end - start} seconds')
  return outputs, targets

