# 完全独立
import torch as t
from itertools import chain
nn = t.nn
import bert_ja as B
import logging
import time
import utils as U
from importlib import reload
import requests
import utils_lite
from self_attention import Multihead_SelfAtt, Multihead_Official, Multihead_Official_Scores
from transformers import BertModel, BertJapaneseTokenizer
import random
import torch.optim as optim
import data_jap_reader as data
import danraku_runner_simple as runner
R = runner
import numpy as np
import mainichi

# Global Variable
GPU_OK = t.cuda.is_available()
G = {} 

# 初始化logging
U.init_logger('spsc.log')

# ==================== 功能函数 ============================

def get_datas(index, epoch, desc, dic_to_send = None, with_dev = True):
  if with_dev:
    return get_datas_org(index, epoch, G['m'], G['ld'], G['testld'], devld = G['devld'], desc = desc, dic_to_send = dic_to_send)
  else:
    return get_datas_org(index, epoch, G['m'], G['ld'], G['testld'], desc = desc, dic_to_send = dic_to_send)

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

def get_datas_org(index, epoch, m, ld, testld, devld = None,  desc='Nothing', dic_to_send = None):
  losses = train_simple(m, ld, epoch) # only one epoch for order matter model
  G[f'testdic_{index}'] = get_test_result_dic(m, testld)
  G[f'losses_{index}'] = losses
  dic = {
    'testdic': G[f'testdic_{index}'],
    'losses': losses
  }
  if devld is not None:
    G[f'devdic_{index}'] = get_test_result_dic(m, devld)
    dic['devdic'] = G[f'devdic_{index}']
  if dic_to_send is not None:
    dic = {**dic, **dic_to_send}
  R.request_my_logger(dic, desc)
  return losses

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
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    dic['prec'] = prec
    dic['rec'] = rec
    dic['f1'] = f1
    dic['bacc'] = bacc
  return dic


def fit_sigmoided_to_label(out):
  assert len(out.shape) == 2
  results = []
  for item in out:
    assert item >= 0 and item <= 1
    if item < 0.5:
      results.append(0) 
    else:
      results.append(1) 
  return t.LongTensor(results)

# ==================== 功能函数 ============================

# 分裂sector, 2vs2的时候，同时判断三个分割点
class Sector_Split(nn.Module):
  def __init__(self, fl_rate = 0, learning_rate = 2e-5):
    super().__init__()
    self.fl_rate = fl_rate
    self.learning_rate = learning_rate
    self.max_memory_batch = 6
    self.bert_size = 768
    self.CEL = nn.CrossEntropyLoss()
    self.verbose = False
    self.init_bert()
    self.init_hook()
    self.optim = optim.AdamW(self.get_should_update(), self.learning_rate)
    print(f'Init AdamW with lr = {self.learning_rate}')
    if GPU_OK:
      _ = self.cuda()

  def handle_mass(self, mass): # 和之前的不一样，这个idea需要保留所有labels
    ss = []
    labels = []
    pos = []
    for s,l,p in mass:
      ss.append(s)
      labels.append(l) # s1 [sep1] s2 [sep2] s3 [sep3] => [sep1, sep2]
      pos.append(p)
    return ss, labels, pos

  def init_bert(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

  def set_verbose(self):
    self.verbose = not self.verbose

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {o.argmax(1).tolist()} Loss: {loss} ')

  # 支持FL
  def cal_loss(self, out, labels, rate = None):
    assert len(labels.shape) == 1
    assert len(out.shape) == 2
    assert labels.shape[0] == out.shape[0]
    rate = self.fl_rate if rate is None else rate
    total = []
    for o, l in zip(out, labels):
      pt = o if (l == 1) else (1 - o)
      loss = (-1) * t.log(pt) * t.pow((1 - pt), rate)
      total.append(loss)
    total = t.stack(total)
    return total.sum()

  def pool_policy(self, ss, pos):
    cls, seps = B.compress_by_ss_pos_get_special_tokens(self.bert, self.toker, ss)

  def init_hook(self): 
    self.classifier = nn.Sequential( # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
      nn.Linear(self.bert_size, 20),
      nn.LeakyReLU(0.1),
      nn.Linear(20, 20),
      nn.LeakyReLU(0.1),
      nn.Linear(20, 1),
      nn.Sigmoid()
    )
    self.classifier2 = nn.Sequential( # 暂时没用上，但是也懒得删掉再加了
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.dry_run_labels = []
    self.dry_run_output = []

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.classifier2.parameters())

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    losses = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss):
      if len(ss) != 4:
        print(f'Warning: less than 4 sentences. {ss[0]}')
      cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
      seps = seps[:-1] # 最后一个SEP不需要
      ls = ls[1:] # 第一个label不需要
      ls = t.LongTensor(ls) # (ss_len), (0 or 1)
      if GPU_OK:
        ls = ls.cuda()
      assert ls.shape[0] == seps.shape[0]
      o = self.classifier(seps) #(ss_len, 1)
      loss = self.cal_loss(o, ls)
      losses.append(loss)
    loss = t.stack(losses).sum()
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    pos_outs = []
    pos_labels = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss):
      if len(ss) != 4:
        print(f'Warning: less than 4 sentences. {ss[0]}')
      cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
      emb = seps[pos - 1] # dry run只需要判断必要部分, (784)
      pos_outs.append(self.classifier(emb).view(1))
      pos_labels.append(ls[pos - 1])
    pos_outs = t.stack(pos_outs)
    pos_labels = t.LongTensor(pos_labels)
    if GPU_OK:
      pos_labels = pos_labels.cuda()
    assert len(pos_outs.shape) == 2 
    assert len(pos_labels.shape) == 1
    assert pos_outs.shape[0] == pos_labels.shape[0]
    self.print_train_info(pos_outs, pos_labels, -1)
    return fit_sigmoided_to_label(pos_outs), pos_labels

# 两边cls中间mean
class Sector_Split2(Sector_Split):
  def init_hook(self): 
    self.classifier = nn.Sequential( # 普通分类
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.classifier2 = nn.Sequential( # 中间分类
      nn.Linear(self.bert_size * 2, 10),
      nn.LeakyReLU(0.1),
      nn.Linear(10, 10),
      nn.LeakyReLU(0.1),
      nn.Linear(10, 1),
      nn.Sigmoid()
    )
    self.dry_run_labels = []
    self.dry_run_output = []

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.classifier2.parameters())

  def cal_loss_return_left(self, ss, label, dry = False):
    assert len(ss) == 2
    cls, seps, sentence_tokens = B.compress_by_ss_pos_get_all_tokens(self.bert, self.toker, ss)
    # 用cls来判断分割点
    assert len(sentence_tokens) == 2
    tokens = sentence_tokens[0] # (?, 784)
    mean = tokens.mean(0)
    if not dry:
      label = t.LongTensor([label]) # (1)
      if GPU_OK:
        label = label.cuda()
      o = self.classifier(cls.view(1, -1)) # (1, 1)
      loss = self.cal_loss(o, label)
      return loss, mean
    else:
      return mean

  def cal_loss_return_right(self, ss, label, dry = False):
    assert len(ss) == 2
    cls, seps, sentence_tokens = B.compress_by_ss_pos_get_all_tokens(self.bert, self.toker, ss)
    # 用cls来判断分割点
    assert len(sentence_tokens) == 2
    tokens = sentence_tokens[1] # (?, 784)
    mean = tokens.mean(0)
    if not dry:
      label = t.LongTensor([label]) # (1)
      if GPU_OK:
        label = label.cuda()
      o = self.classifier(cls.view(1, -1)) # (1, 1)
      loss = self.cal_loss(o, label)
      return loss, mean
    else:
      return mean

  def cal_loss_middle(self, mean_left, mean_right, label, dry = False):
    cat_emb = t.cat((mean_left, mean_right)).view(1, -1) # (1, 2 * 784)
    assert cat_emb.shape[1] == self.bert_size * 2
    o = self.classifier2(cat_emb) # (1, 1)
    if not dry:
      label = t.LongTensor([label]) # (1)
      if GPU_OK:
        label = label.cuda()
      loss = self.cal_loss(o, label)
      return loss
    else:
      return o

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    losses = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss):
      if len(ss) != 4: # 不训练
        print(f'Warning: less than 4 sentences. {ss[0]}')
        pass
      else:
        left_loss, left_mean = self.cal_loss_return_right([ss[0], ss[1]], ls[1]) # s1 = l0, s2 = l1, s3 = l2, s4 = l3
        right_loss, right_mean = self.cal_loss_return_left([ss[2], ss[3]], ls[3]) # s1 = l0, s2 = l1, s3 = l2, s4 = l3
        middle_loss = self.cal_loss_middle(left_mean, right_mean, ls[2])
        loss = left_loss + right_loss + middle_loss
        losses.append(loss)
    if len(losses) < 1:
      return 0
    else:
      loss = t.stack(losses).sum()
      self.zero_grad()
      loss.backward()
      self.optim.step()
      # self.print_train_info(o, labels, loss.detach().item())
      return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    pos_outs = []
    pos_labels = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss):
      if len(ss) != 4: # 不训练
        print(f'Warning: less than 4 sentences. {ss[0]}')
        pass
      else:
        left_mean = self.cal_loss_return_left([ss[0], ss[1]], ls[1], dry = True) # s1 = l0, s2 = l1, s3 = l2, s4 = l3
        right_mean = self.cal_loss_return_right([ss[2], ss[3]], ls[3], dry = True)
        o = self.cal_loss_middle(left_mean, right_mean, ls[2], dry = True)
        pos_outs.append(o.view(1))
        pos_labels.append(ls[2])
    if len(pos_outs) < 1:
      return t.LongTensor([]), t.LongTensor([])
    else:
      pos_outs = t.stack(pos_outs)
      pos_labels = t.LongTensor(pos_labels)
      if GPU_OK:
        pos_labels = pos_labels.cuda()
      assert len(pos_outs.shape) == 2 
      assert len(pos_labels.shape) == 1
      assert pos_outs.shape[0] == pos_labels.shape[0]
      self.print_train_info(pos_outs, pos_labels, -1)
      return fit_sigmoided_to_label(pos_outs), pos_labels

# 两边cls中间也cls
class Sector_Split3(Sector_Split2):
  def cal_loss_return_left(self, ss, label, dry = False):
    assert len(ss) == 2
    cls, seps, sentence_tokens = B.compress_by_ss_pos_get_all_tokens(self.bert, self.toker, ss)
    # 用cls来判断分割点
    assert len(sentence_tokens) == 2
    tokens = sentence_tokens[0] # (?, 784)
    mean = tokens.mean(0)
    if not dry:
      label = t.LongTensor([label]) # (1)
      if GPU_OK:
        label = label.cuda()
      o = self.classifier(cls.view(1, -1)) # (1, 1)
      loss = self.cal_loss(o, label)
      return loss, cls
    else:
      return cls

  def cal_loss_return_right(self, ss, label, dry = False):
    assert len(ss) == 2
    cls, seps, sentence_tokens = B.compress_by_ss_pos_get_all_tokens(self.bert, self.toker, ss)
    # 用cls来判断分割点
    assert len(sentence_tokens) == 2
    tokens = sentence_tokens[1] # (?, 784)
    mean = tokens.mean(0)
    if not dry:
      label = t.LongTensor([label]) # (1)
      if GPU_OK:
        label = label.cuda()
      o = self.classifier(cls.view(1, -1)) # (1, 1)
      loss = self.cal_loss(o, label)
      return loss, cls
    else:
      return cls

# =============================== Model ===========================

def init_G_Symmetry_Mainichi(half = 1, batch = 4, mini = False):
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_trains(mini))
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_tests(mini))
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

def run_old():
  init_G_Symmetry_Mainichi(half = 2, batch = 2)
  for i in range(5):
    G['m'] = m = Sector_Split(learning_rate = 5e-6)
    get_datas(0, 1, f'分裂sector E1', with_dev = False)
    get_datas(0, 1, f'分裂sector E2', with_dev = False)
    get_datas(0, 1, f'分裂sector E3', with_dev = False)
  

def run():
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  for i in range(20):
    G['m'] = m = Sector_Split2(learning_rate = 5e-6)
    get_datas(i, 2, f'Sector_Split2', with_dev = False)
    G['m'] = m = Sector_Split3(learning_rate = 5e-6)
    get_datas(i + 100, 2, f'Sector_Split3', with_dev = False)

