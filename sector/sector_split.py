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
G = {
  'mess_list': [] # 用来存储结果，不用每次都从网上扒累死了
} 

# 初始化logging
U.init_logger('spsc.log')

# ==================== 功能函数 ============================

def get_datas(index, epoch, desc, dic_to_send = None, with_dev = True, url = None):
  if with_dev:
    return get_datas_org(index, epoch, G['m'], G['ld'], G['testld'], devld = G['devld'], desc = desc, dic_to_send = dic_to_send, url = url)
  else:
    return get_datas_org(index, epoch, G['m'], G['ld'], G['testld'], desc = desc, dic_to_send = dic_to_send, url = url)


def train_simple(m, loader, epoch):
  logger = logging.debug
  loss_per_epoch = []
  if hasattr(loader, 'start'):
    loader.start = 0
  start = time.time()
  if hasattr(loader, 'ds'):
    length = len(loader.ds.datas)
  else:
    length = len(loader)
  for e in range(epoch):
    if hasattr(loader, 'shuffle'):
      loader.shuffle()
    else:
      random.shuffle(loader)
    logger(f'start epoch{e}')
    total_loss = 0
    index = 0
    for mass in loader:
      index += 1
      logger(f'{index}/{length}')
      total_loss += m.train(mass)
    avg_loss = total_loss / length
    loss_per_epoch.append(total_loss)
  end = time.time()
  if hasattr(loader, 'batch_size'):
    batch_size = loader.batch_size
  else:
    batch_size = 'unknown'
  logger(f'Trained! Epochs: {epoch}, Batch size: {batch_size}, dataset length: {length}, Time cost: {end - start} seconds')
  return loss_per_epoch

def get_datas_org(index, epoch, m, ld, testld, devld = None,  desc='Nothing', dic_to_send = None, url = None):
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
  G['mess_list'].append(dic) # 存到mess里到时直接复制到结果
  R.request_my_logger(dic, desc, url)
  return losses

def get_datas_early_stop(index, epochs, desc, dic_to_send = None, url = None):
  valid_losses = []
  train_losses = []
  tested = []
  for i in range(epochs):
    train_losses += train_simple(G['m'], G['ld'], 1)
    valid_loss = cal_total_loss(G['m'], G['validld'])
    valid_losses.append(valid_loss)
    dic_to_analyse = get_test_result_dic(G['m'], G['testld'])
    dic_to_analyse['index'] = i # Save index info
    dic_to_analyse['valid_loss'] = valid_loss # Save index info
    tested.append(dic_to_analyse)
  test_result = tested[np.argmin(valid_losses)]
  # test_result['valid_losses'] = valid_losses
  G['mess_list'].append(test_result) # 将valid loss最小对应的dic放进mess_list
  dic = test_result
  if dic_to_send is not None:
    dic = {**dic, **dic_to_send}
  else:
    pass
  R.request_my_logger(dic, desc, url)

def get_test_result(m, loader):
  logger = logging.debug
  if hasattr(loader, 'start'):
    loader.start = 0
  start = time.time()
  if hasattr(loader, 'ds'):
    length = len(loader.ds.datas)
  else:
    length = len(loader)
  outputs = []
  targets = []
  index = 0
  for mass in loader:
    index += 1
    logger(f'TESTING: {index}/{length}')
    out, labels = m.dry_run(mass)
    outputs += out.tolist()
    targets += labels.tolist()
  end = time.time()
  logger(f'TESTED! length={length} Time cost: {end - start} seconds')
  return outputs, targets

def get_test_result_dic(m, loader):
  if hasattr(loader, 'start'):
    loader.start = 0
  dic = {}
  if loader is None:
    dic['prec'] = -1
    dic['rec'] = -1
    dic['f1'] = -1
    dic['bacc'] = -1
  else: 
    outputs, targets = get_test_result(m, loader)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    dic['prec'] = prec
    dic['rec'] = rec
    dic['f1'] = f1
    dic['bacc'] = bacc
  return dic

def cal_total_loss(m, loader):
  logger = logging.debug
  if hasattr(loader,'start'):
    loader.start = 0
  start = time.time()
  if hasattr(loader, 'ds'):
    length = len(loader.ds.datas)
  else:
    length = len(loader)
  loss = 0
  for mass in loader:
    # logger(f'VALID: {loader.start}/{length}')
    losses = m.get_loss(mass)
    if len(losses) < 1:
      loss += 0
    else:
      loss += t.stack(losses).detach().sum().item()
  end = time.time()
  logger(f'VALIDED! length={length} Time cost: {end - start} seconds')
  return loss

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

def fit_sigmoided_to_label_list(out):
  results = []
  for item in out:
    assert item >= 0 and item <= 1
    if item < 0.5:
      results.append(0) 
    else:
      results.append(1) 
  return results

# ==================== 功能函数 ============================

# 分裂sector, 2vs2的时候，同时判断三个分割点
class Sector_Split(nn.Module):
  def __init__(self, fl_rate = 0, learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0):
    super().__init__()
    self.ss_len_limit = ss_len_limit
    self.fl_rate = fl_rate
    self.auxiliary_loss_rate = auxiliary_loss_rate
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
    # print(rate)
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

  def get_loss(self, mass): 
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    loss = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      if len(ss) != self.ss_len_limit: # 直接不训练
        #print(f'Warning: less than {self.ss_len_limit} sentences. {ss[0]}')
        pass
      else: 
        cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
        seps = seps[:-1] # 最后一个SEP不需要
        ls = ls[1:] # 第一个label不需要
        ls = t.LongTensor(ls) # (ss_len), (0 or 1)
        if GPU_OK:
          ls = ls.cuda()
        assert ls.shape[0] == seps.shape[0]
        # 稍微做一点tricky的事情，将其他loss(除了中间那个) * 0.5
        o = self.classifier(seps) #(ss_len, 1)
        for index, (o_item, l_item) in enumerate(zip(o, ls)): # Different split point, only middle important
          loss_part = self.cal_loss(o_item.view(1, 1), l_item.view(1))
          if index != int((len(ls) / 2)):
            loss_part = loss_part * self.auxiliary_loss_rate
          else:
            assert index == int(len(ss) / 2) - 1 == pos - 1
            pass
          loss.append(loss_part)
    return loss

  def train(self, mass):
    losses = self.get_loss(mass)
    if len(losses) < 1:
      return 0
    else:
      loss = t.stack(losses).sum()
      self.zero_grad()
      loss.backward()
      self.optim.step()
      return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    pos_outs = []
    pos_labels = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      if len(ss) != self.ss_len_limit: # 直接不跑
        #print(f'Warning: less than {self.ss_len_limit} sentences. {ss[0]}')
        pass
      else: 
        cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
        seps = seps[:-1] # 最后一个SEP不需要
        ls = ls[1:] # 第一个label不需要
        ls = t.LongTensor(ls) # (ss_len), (0 or 1)
        if GPU_OK:
          ls = ls.cuda()
        assert ls.shape[0] == seps.shape[0]
        emb = seps[pos - 1] # 取中间那个
        pos_outs.append(self.classifier(emb).view(1))
        pos_labels.append(ls[pos - 1])
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
        #print(f'Warning: less than 4 sentences. {ss[0]}')
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
        #print(f'Warning: less than 4 sentences. {ss[0]}')
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


# @Deprecated
# 千万别用这个，这个是派生用的
class Sector_Standard(Sector_Split):
  def get_pooled(self, ss, pos):
    cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
    emb = seps[pos] # s0 s1 s2 s3, (784)
    assert len(ls) == len(seps)
    return emb

  def get_loss(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    losses = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss):
      if len(ss) != self.ss_len_limit: # 不训练
        #print(f'Warning: No Train less with ss_len = {len(ss)}. {ss[0]}')
        pass
      else:
        assert pos == self.ss_len_limit / 2
        emb = self.get_pooled(ss, pos)
        o = self.classifier(emb).view(1, 1) # (1,1)
        l = ls[pos]
        l = t.LongTensor([l]) # (1)
        if GPU_OK:
          l = l.cuda()
        loss = self.cal_loss(o, l)
        losses.append(loss)
    return losses

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    pos_outs = []
    pos_labels = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss):
      if len(ss) != self.ss_len_limit: # 不训练
        #print(f'Warning: No Train less with ss_len = {len(ss)}. {ss[0]}')
        pass
      else:
        assert pos == self.ss_len_limit / 2
        emb = self.get_pooled(ss, pos)
        o = self.classifier(emb).view(1) # (1,1)
        pos_outs.append(o)
        pos_labels.append(ls[pos])
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
      return fit_sigmoided_to_label(pos_outs), pos_labels
  

class Sector_Standard_Many_SEP_Pool_CLS(Sector_Standard):
  def get_pooled(self, ss, pos):
    cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
    assert len(ls) == len(seps)
    emb = cls # s0 s1 s2 s3, (784)
    return emb

class Sector_Standard_One_SEP_One_CLS_Pool_CLS(Sector_Standard):
  def get_pooled(self, ss, pos):
    cls = B.compress_one_cls_one_sep_pool_cls(self.bert, self.toker, ss, pos)
    return cls


class Sector_Standard_Many_SEP(Sector_Standard):
  def get_pooled(self, ss, pos):
    cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
    assert len(seps) == len(ss)
    return seps[pos - 1]

class Sector_Plus_Ordering(Sector_Split): 
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    loss = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      if len(ss) != self.ss_len_limit: # 直接不训练
        #print(f'Warning: less than {self.ss_len_limit} sentences. {ss[0]}')
        pass
      else: 
        cls, sep_middle = B.compress_by_ss_get_cls_and_middle_sep(self.bert, self.toker, ss)
        label = t.LongTensor([ls[pos]]).view(1) # (ss_len), (0 or 1)
        if GPU_OK:
          label = label.cuda()
        sector_loss = self.cal_loss(self.classifier(sep_middle).view(1,1), label)
        # 计算order loss 
        ss_shuffled = ss.copy()
        if random.random() > 0.5: # 50%交换中间两句话
          ss_shuffled[1], ss_shuffled[2] = ss_shuffled[2], ss_shuffled[1]
          label_order = t.LongTensor([1]).view(1)
        else:
          label_order = t.LongTensor([0]).view(1)
        cls_shuffled, _ = B.compress_by_ss_get_cls_and_middle_sep(self.bert, self.toker, ss_shuffled)
        if GPU_OK:
          label_order = label_order.cuda()
        order_loss = self.cal_loss(self.classifier2(cls_shuffled).view(1,1), label_order)
        loss.append(sector_loss + self.auxiliary_loss_rate * order_loss)
    if len(loss) < 1:
      return 0
    else:
      loss = t.stack(loss).sum()
      self.zero_grad()
      loss.backward()
      self.optim.step()
      return loss.detach().item()

# ======= BiLSTM

class Split_GRU(Sector_Split):
  def init_hook(self): 
    self.gru = nn.GRU(input_size=self.bert_size, hidden_size= int(self.bert_size / 2), batch_first = True, bidirectional = True) # in: (1, seq_len, 768) out:  (1, seq_len, 768)
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
    return chain(self.bert.parameters(), self.classifier.parameters(), self.classifier2.parameters(), self.gru.parameters())

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    loss = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      if len(ss) != self.ss_len_limit: # 直接不训练
        #print(f'Warning: less than {self.ss_len_limit} sentences. {ss[0]}')
        pass
      else: 
        clss = t.stack([B.compress_one_cls_one_sep_pool_cls(self.bert, self.toker, [s], 0) for s in ss]) # (seq_len, 768)
        # Put in context using bi-gru
        clss = clss.view(1, len(ss), self.bert_size)
        clss, _ = self.gru(clss) 
        clss = clss.view(len(ss), self.bert_size)
        clss = clss[1:] # 第一个cls不需要
        # Process labels
        ls = ls[1:] # 第一个label不需要
        ls = t.LongTensor(ls) # (ss_len), (0 or 1)
        if GPU_OK:
          ls = ls.cuda()
        assert ls.shape[0] == clss.shape[0]
        # 稍微做一点tricky的事情，将其他loss(除了中间那个) * 0.5
        o = self.classifier(clss) #(ss_len, 1)
        for index, (o_item, l_item) in enumerate(zip(o, ls)): # Different split point, only middle important
          loss_part = self.cal_loss(o_item.view(1, 1), l_item.view(1))
          if index != int((len(ls) / 2)):
            loss_part = loss_part * self.auxiliary_loss_rate
          else:
            assert index == int(len(ss) / 2) - 1 == pos - 1
            pass
          loss.append(loss_part)
    if len(loss) < 1:
      return 0
    else:
      loss = t.stack(loss).sum()
      self.zero_grad()
      loss.backward()
      self.optim.step()
      return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    pos_outs = []
    pos_labels = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      if len(ss) != self.ss_len_limit: # 直接不跑
        #print(f'Warning: less than {self.ss_len_limit} sentences. {ss[0]}')
        pass
      else: 
        clss = t.stack([B.compress_one_cls_one_sep_pool_cls(self.bert, self.toker, [s], 0) for s in ss]) # (seq_len, 768)
        # Put in context using bi-gru
        clss = clss.view(1, len(ss), self.bert_size)
        clss, _ = self.gru(clss) 
        clss = clss.view(len(ss), self.bert_size)
        clss = clss[1:] # 第一个cls不需要
        # labels processing
        ls = ls[1:] # 第一个label不需要
        ls = t.LongTensor(ls) # (ss_len), (0 or 1)
        if GPU_OK:
          ls = ls.cuda()
        assert ls.shape[0] == clss.shape[0]
        emb = clss[pos - 1] # 取中间那个
        pos_outs.append(self.classifier(emb).view(1))
        pos_labels.append(ls[pos - 1])
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
      return fit_sigmoided_to_label(pos_outs), pos_labels


class GRU_Standard(Split_GRU):
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    loss = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      if len(ss) != self.ss_len_limit: # 直接不训练
        #print(f'Warning: less than {self.ss_len_limit} sentences. {ss[0]}')
        pass
      else: 
        clss = t.stack([B.compress_one_cls_one_sep_pool_cls(self.bert, self.toker, [s], 0) for s in ss]) # (seq_len, 768)
        # Put in context using bi-gru
        clss = clss.view(1, len(ss), self.bert_size)
        clss, _ = self.gru(clss) 
        clss = clss.view(len(ss), self.bert_size)
        clss = clss[1:] # 第一个cls不需要
        # Process labels
        ls = ls[1:] # 第一个label不需要
        ls = t.LongTensor(ls) # (ss_len), (0 or 1)
        if GPU_OK:
          ls = ls.cuda()
        assert ls.shape[0] == clss.shape[0]
        # 稍微做一点tricky的事情，将其他loss(除了中间那个) * 0.5
        l_item = ls[int((len(ls) / 2))]
        o_item = clss[int((len(ls) / 2))]
        o_item = self.classifier(o_item.view(1, self.bert_size))
        loss_part = self.cal_loss(o_item.view(1, 1), l_item.view(1))
        loss.append(loss_part)
    if len(loss) < 1:
      return 0
    else:
      loss = t.stack(loss).sum()
      self.zero_grad()
      loss.backward()
      self.optim.step()
      return loss.detach().item()

# =============================== Model ===========================

def init_G_Symmetry_Mainichi(half = 1, batch = 4, mini = False):
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_trains(mini))
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_tests(mini))
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

def init_G_Symmetry_Mainichi_With_Valid(half = 1, batch = 4, mini = False):
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_trains(mini))
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_tests(mini))
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_valid(mini))
  G['validld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

def init_G_Symmetry_Mainichi_big(half = 1, batch = 4, mini = False):
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_trains_big(mini))
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_tests_big(mini))
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

def init_G_Symmetry_Syosetu(half = 1, batch = 4):
  ds = data.Dataset(ss_len = half * 2, datas = data.read_trains())
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = data.read_tests())
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

# =============================== Dataset =========================



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
    get_datas(i, 2, f'Sector_Split2 mean', with_dev = False)
    G['m'] = m = Sector_Split3(learning_rate = 5e-6)
    get_datas(i + 100, 2, f'Sector_Split3 cls', with_dev = False)

# 用于测试多个sep是不是会掉精度
def run_standard():
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  # G['m'] = m = Sector_Standard(learning_rate = 5e-6)
  for i in range(20):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas(i, 2, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 2', with_dev = False)
  
def run_1vs1():
  init_G_Symmetry_Mainichi(half = 1, batch = 2, mini = False)
  for i in range(20):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 2e-5, ss_len_limit = 2)
    get_datas(i, 2, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 2', with_dev = False)

def run_many_seps():
  for i in range(20):
    init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
    G['m'] = m = Sector_Standard_Many_SEP(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas(i + 100, 2, f'Sector_Standard_Many_SEP 2vs2 2', with_dev = False)
    init_G_Symmetry_Mainichi(half = 1, batch = 2, mini = False)
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 2e-5, ss_len_limit = 2)
    get_datas(i, 2, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 2', with_dev = False)

def run_split_with_auxiliary_rate(auxiliary_loss_rate = 0.8):
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  for i in range(15):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = auxiliary_loss_rate)
    get_datas(i, 2, f'Sector_Split with auxiliary rate {m.auxiliary_loss_rate} 2vs2 2', with_dev = False)


def run_sector_ordering(auxiliary_loss_rate = 0.5):
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  for i in range(15):
    G['m'] = m = Sector_Plus_Ordering(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = auxiliary_loss_rate)
    get_datas(i, 2, f'Sector_Plus_Ordering with auxiliary rate {m.auxiliary_loss_rate} 2vs2 2', with_dev = False)

def run():
  run_sector_ordering()
  run_split_with_auxiliary_rate(0.8)

def run_3vs3_standard_split():
  init_G_Symmetry_Mainichi(half = 3, batch = 2, mini = False)
  # G['m'] = m = Sector_Standard_Many_SEP(learning_rate = 5e-7, ss_len_limit = 6)
  for i in range(15):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 6, auxiliary_loss_rate = 0.5)
    get_datas(i, 2, f'NO{i} Sector_Split 3vs3 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False)
    get_datas(i + 20, 1, f'NO{i} Sector_Split 3vs3 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False)
  # for i in range(15):
  #   G['m'] = m = Sector_Standard_Many_SEP(learning_rate = 5e-7, ss_len_limit = 6)
  #   get_datas(i, 2, f'Sector_Standard_Many_SEP 3vs3 2', with_dev = False)
  #   G['m'] = m = Sector_Split(learning_rate = 5e-7, ss_len_limit = 6, auxiliary_loss_rate = 0.5)
  #   get_datas(i+20, 2, f'Sector_Split with auxiliary rate {m.auxiliary_loss_rate} 3vs3 2', with_dev = False)

def run_2vs2_standard():
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  for i in range(18):
    G['m'] = m = Sector_Standard_Many_SEP(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas(i, 2, f'NO.{i} Sector_Standard_Many_SEP 2vs2 2', with_dev = False)
    get_datas(i + 20, 1, f'NO.{i} Sector_Standard_Many_SEP 2vs2 3', with_dev = False)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.5)
    get_datas(i + 100, 2, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False)
    get_datas(i + 120, 1, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False)

def auxiliary_rate_test():
  panther_url = 'https://hookb.in/Z2dBDBMP39uR33eLJXYO'
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  for i in range(10):
    # Rate = 0.1
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.1)
    get_datas(0, 2, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url) 
    # Rate = 0.2
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.2)
    get_datas(0, 2, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url) 
    # Rate = 0.8
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.8)
    get_datas(0, 2, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url) 
    # Rate = 1
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 1.0)
    get_datas(0, 2, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url) 
    # Rate = 1.5
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 1.5)
    get_datas(0, 2, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO.{i} Sector_Split 2vs2 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url) 

def run_3vs3():
  init_G_Symmetry_Mainichi(half = 3, batch = 2, mini = False)
  panther_url = 'https://hookb.in/Z2dBDBMP39uR33eLJXYO'
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 6, auxiliary_loss_rate = 0.5)
    get_datas(0, 2, f'NO{i} Sector_Split 3vs3 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO{i} Sector_Split 3vs3 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url)

def run_4vs4():
  init_G_Symmetry_Mainichi(half = 4, batch = 2, mini = False)
  panther_url = 'https://hookb.in/Z2dBDBMP39uR33eLJXYO'
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 8, auxiliary_loss_rate = 0.5)
    get_datas(0, 2, f'NO{i} Sector_Split 4vs4 auxiliary rate {m.auxiliary_loss_rate} 2', with_dev = False, url = panther_url)
    get_datas(0, 1, f'NO{i} Sector_Split 4vs4 auxiliary rate {m.auxiliary_loss_rate} 3', with_dev = False, url = panther_url)


def run_2vs2_1cls_1sep_optimal():
  init_G_Symmetry_Mainichi(half = 2, batch = 2, mini = False)
  # G['m'] = m = Sector_Standard(learning_rate = 5e-6)
  for i in range(20):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas(i, 2, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 2', with_dev = False)
    get_datas(i + 100, 1, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 3', with_dev = False)

def run_syosetu_my_computer():
  panther_url = 'https://hookb.in/BYj9ewaWLwhLDDx31JrN'
  init_G_Symmetry_Syosetu(half = 1, batch = 4)
  for i in range(10):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 2)
    get_datas(i, 2, f'小说Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 2', with_dev = False, url = panther_url)
    get_datas(i + 100, 1, f'小说Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 3', with_dev = False, url = panther_url)
  init_G_Symmetry_Syosetu(half = 2, batch = 2)
  for i in range(10):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas(i, 2, f'小说Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 2', with_dev = False, url = panther_url)
    get_datas(i + 100, 1, f'小说Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 3', with_dev = False, url = panther_url)
  for i in range(10):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.5)
    get_datas(i, 2, f'小说Sector_Split 2vs2 0.5 2', with_dev = False, url = panther_url)
    get_datas(i + 100, 1, f'小说Sector_Split 2vs2 0.5 3', with_dev = False, url = panther_url)
  for i in range(10):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.2)
    get_datas(i, 2, f'小说Sector_Split 2vs2 0.2 2', with_dev = False, url = panther_url)
    get_datas(i + 100, 1, f'小说Sector_Split 2vs2 0.2 3', with_dev = False, url = panther_url)
     

def run_syosetu_panther():
  panther_url = 'https://hookb.in/aBe1JqzJjQf1oobLKD0E'
  init_G_Symmetry_Syosetu(half = 3, batch = 2)
  for i in range(10):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 6, auxiliary_loss_rate = 0.5)
    get_datas(i, 2, f'小说Sector_Split 3vs3 0.5 2', with_dev = False, url = panther_url)
    get_datas(i + 100, 1, f'小说Sector_Split 3vs3 0.5 3', with_dev = False, url = panther_url)
  init_G_Symmetry_Syosetu(half = 4, batch = 2)
  for i in range(10):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 8, auxiliary_loss_rate = 0.5)
    get_datas(i, 2, f'小说Sector_Split 4vs4 0.5 2', with_dev = False, url = panther_url)
    get_datas(i + 100, 1, f'小说Sector_Split 4vs4 0.5 3', with_dev = False, url = panther_url)

def run_lstm():
  panther_url = 'https://hookb.in/OeMdR0z87NinzzlkeNMj'
  init_G_Symmetry_Mainichi(half = 2, batch = 2)
  for i in range(20):
    G['m'] = m = Split_GRU(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.5)
    # G['m'] = m = GRU_Standard(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.5)
    get_datas(i, 1, f'GRU 2vs2 0.5 1', with_dev = False, url = panther_url)
    get_datas(i, 1, f'GRU 2vs2 0.5 2', with_dev = False, url = panther_url)
    get_datas(i, 1, f'GRU 2vs2 0.5 3', with_dev = False, url = panther_url)

# TODO: 记录的时候要分开
def run_big():
  panther_url = 'https://hookb.in/jekwxj6je2teBB23OK8y'
  init_G_Symmetry_Mainichi_big(half = 2, batch = 2)
  for i in range(20):
    G['m'] = m = Sector_Standard_Many_SEP(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas(i + 200, 1, f'BIG Sector_Standard_Many_SEP 2vs2 1', with_dev = False, url = panther_url)
    get_datas(i + 200, 1, f'BIG Sector_Standard_Many_SEP 2vs2 2', with_dev = False, url = panther_url)
    get_datas(i + 200, 1, f'BIG Sector_Standard_Many_SEP 2vs2 3', with_dev = False, url = panther_url)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.5)
    get_datas(i, 1, f'BIG Sector_Split 2vs2 0.5 1', with_dev = False, url = panther_url)
    get_datas(i, 1, f'BIG Sector_Split 2vs2 0.5 2', with_dev = False, url = panther_url)
    get_datas(i, 1, f'BIG Sector_Split 2vs2 0.5 3', with_dev = False, url = panther_url)

def run_standard_early_stop():
  panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
  init_G_Symmetry_Mainichi_With_Valid(half = 2, batch = 4, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas_early_stop(i, 3, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 2vs2 early_stop', url = panther_url)

def run_standard1_early_stop():
  panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
  init_G_Symmetry_Mainichi_With_Valid(half = 1, batch = 4, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 2)
    get_datas_early_stop(i, 3, f'Sector_Standard_One_SEP_One_CLS_Pool_CLS 1vs1 early_stop', url = panther_url)

def run_standard3_early_stop():
  panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
  init_G_Symmetry_Mainichi_With_Valid(half = 3, batch = 2, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Standard_One_SEP_One_CLS_Pool_CLS(learning_rate = 5e-6, ss_len_limit = 6)
    get_datas_early_stop(i, 3, f'Standard_One_SEP_One_CLS_Pool_CLS 3vs3 early_stop', url = panther_url)
    
def run_split_early_stop():
  panther_url = 'https://hookb.in/6JzYX9a69jSLbb031pgZ'
  init_G_Symmetry_Mainichi_With_Valid(half = 2, batch = 4, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.5)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)

def run_split_early_stop_3vs3_rate02():
  panther_url = 'https://hookb.in/2qJ68WOWa0H9BBKG1xBj'
  init_G_Symmetry_Mainichi_With_Valid(half = 3, batch = 2, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 6, auxiliary_loss_rate = 0.2)
    get_datas_early_stop(i, 3, f'Sector_Split 3vs3 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)

def run_split_early_stop_4vs4_rate02():
  panther_url = 'https://hookb.in/2qJ68WOWa0H9BBKG1xBj'
  init_G_Symmetry_Mainichi_With_Valid(half = 4, batch = 2, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 8, auxiliary_loss_rate = 0.2)
    get_datas_early_stop(i, 3, f'Sector_Split 4vs4 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)


def run_split_early_stop_rates_test():
  panther_url = 'https://hookb.in/6JzYX9a69jSLbb031pgZ'
  init_G_Symmetry_Mainichi_With_Valid(half = 2, batch = 4, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.2)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.8)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 1.0)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 1.5)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)

def run_split_early_stop_rates_test2():
  panther_url = 'https://hookb.in/6JzYX9a69jSLbb031pgZ'
  init_G_Symmetry_Mainichi_With_Valid(half = 2, batch = 4, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.1)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.3)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)
    G['m'] = m = Sector_Split(learning_rate = 5e-6, ss_len_limit = 4, auxiliary_loss_rate = 0.4)
    get_datas_early_stop(i, 3, f'Sector_Split 2vs2 rate={m.auxiliary_loss_rate} early_stop', url = panther_url)

def run_split_early_stop_rates_zero():
  panther_url = 'https://hookb.in/VGERm7dJyjtE22bwzZ7d'
  init_G_Symmetry_Mainichi_With_Valid(half = 2, batch = 4, mini=False)
  for i in range(20):
    G['m'] = m = Sector_Standard_Many_SEP(learning_rate = 5e-6, ss_len_limit = 4)
    get_datas_early_stop(i, 3, f'Sector_Standard_Many_SEP 2vs2 early_stop', url = panther_url)

