import numpy as np
import torch as t
from importlib import reload
import torch.optim as optim
from itertools import chain
import logging
nn = t.nn
import danraku_runner_simple as R

import data_jap_reader as D
import word2vec_fucker as W

GPU_OK = t.cuda.is_available()

# ===============

class Dataset(t.utils.data.dataset.Dataset):
  def __init__(self, ss_len = 8, max_ids = 64):
    super().__init__()
    self.ss_len = ss_len
    self.max_ids = max_ids
    self.init_datas_hook()
    self.init_hook()
    self.start = 0

  def init_hook(self):
    pass

  def init_datas_hook(self):
    self.datas = []

  def set_datas(self, datas):
    self.datas = datas

  def is_begining(self, s):
    return s.startswith('\u3000')
        
  def no_indicator(self, ss):
    return [s.replace('\u3000', '') for s in ss]

  # 作为最底层的方法，需要保留所有分割信息
  def get_ss_and_labels(self, start): 
    end = min(start + self.ss_len, len(self.datas)) 
    ss = []
    labels = []
    for i in range(start, end):
      s = self.datas[i]
      labels.append(1 if self.is_begining(s) else 0)
      ss.append(s)
    ss = self.no_indicator(ss)
    return ss, labels

  def __getitem__(self, start):
    return self.get_ss_and_labels(start)

  def __len__(self):
    return len(self.datas)

  def shuffle(self):
    random.shuffle(self.datas)


class DatasetPos(Dataset):
  # 作为最底层的方法，需要保留所有分割信息
  def get_ss_and_labels(self, cut_point): 
    half = int(self.ss_len / 2)
    start = cut_point - half
    start = max(start, 0)
    end = cut_point + half # 因为在range的右边，所以没必要-1
    end = min(end, len(self.datas))
    ss = self.no_indicator([self.datas[i] for i in range(start, end)])
    label = 1 if self.is_begining(self.datas[cut_point]) else 0
    pos_relative = cut_point - start
    return ss, label, pos_relative

  def __getitem__(self, start):
    ss, label, pos_relative = self.get_ss_and_labels(start)
    return ss, label, pos_relative


def w2v(ss, max_len):
  results = []
  for s in ss:
    wordvecs = W.sentence_to_wordvecs(s, max_len)
    if len(wordvecs) > 0:
      results.append(t.tensor(wordvecs))
    else:
      print('Waring: sentence to a empty wordvecs')
      results.append(t.zeros(1, 300))
  return results

class Loader_Pos():
  def __init__(self, ds, max_len = 64):
    self.ds = ds
    self.dataset = ds
    self.start = 0
    self.batch_size = self.ds.ss_len
    self.max_len = max_len

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.ds.datas)

  def __next__(self):
    if self.start >= len(self):
      self.start = 0
      raise StopIteration()
    else:
      ss, label, pos_relative = self.ds[self.start]
      self.start += 1
      ss_tensor = w2v(ss, self.max_len) # (seq_len, ?, 300)
      ss_padded = t.nn.utils.rnn.pad_sequence(ss_tensor, True) # (seq_len, max_token, 300)
      return ss_padded, (t.LongTensor([label]), t.LongTensor([pos_relative]))

  def shuffle(self):
    self.ds.shuffle()

def ld(ss_len = 2, max_len = 64):
  ds = DatasetPos(ss_len = ss_len)
  ds.set_datas(D.read_trains())
  return Loader_Pos(ds, max_len)

def tld(ss_len = 2, max_len = 64):
  ds = DatasetPos(ss_len = ss_len)
  ds.set_datas(D.read_tests())
  ld = Loader_Pos(ds, max_len)
  return Loader_Pos(ds, max_len)

def dld(ss_len = 2, max_len = 64):
  ds = DatasetPos(ss_len = ss_len)
  ds.set_datas(D.read_devs())
  return Loader_Pos(ds, max_len)


class WikiSector(nn.Module):
  def __init__(self, hidden_size = 256, weight_one = 1,  head = 8, dropout=0):
    super().__init__()
    self.max_memory_batch = 6
    self.hidden_size = hidden_size
    self.head = head
    self.dropout = dropout
    self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, weight_one]))
    self.verbose = False
    self.wordvec_size = 300
    self.init_hook()
    self.optim = optim.AdamW(self.get_should_update(), self.learning_rate())
    print(f'Init AdamW with lr = {self.learning_rate()}')
    if GPU_OK:
      _ = self.cuda()

  def learning_rate(self):
    return 1e-4

  def init_hook(self):
    self.gru_batch_first_word_compressor = t.nn.GRU(self.wordvec_size, self.hidden_size, batch_first=True)
    self.bi_gru_batch_first_integrator = t.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
    self.classifier = nn.Sequential(
      nn.Linear(self.hidden_size * 2, self.hidden_size),
      nn.LeakyReLU(0.1),
      nn.Linear(self.hidden_size, 2),
    )

  def set_verbose(self):
    self.verbose = not self.verbose

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {o.argmax(1).tolist()} Loss: {loss} ')

  def get_should_update(self):
    return chain(self.bi_gru_batch_first_integrator.parameters(), self.classifier.parameters(), self.gru_batch_first_word_compressor.parameters())

  # inpts: (seq_len, token_len, feature)
  # return: (seq_len, hidden_size)
  def cls(self, inpts):
    seq_len, words, feature = inpts.shape
    _, hn = self.gru_batch_first_word_compressor(inpts) # (1, seq_len, hidden_size)
    return hn.view(seq_len, self.hidden_size)

  def processed_embs(self, embs):
    return embs

  # ss: (seq_len, hidden_size)
  # return: (seq_len, hidden_size * 2)
  def integrate_sentences_info(self, ss):
    seq_len, hidden_size = ss.shape
    ss = ss.view(1, seq_len, hidden_size)
    out, _ = self.bi_gru_batch_first_integrator(ss) # (seq_len, 1, hidden_size * 2)
    return out.view(seq_len, hidden_size * 2)

  # inpts: (seq_len, token_len, feature)
  # labels: (label, pos)
  def train(self, inpts, labels):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = inpts.cuda()
      label = label.cuda()
    embs = self.cls(inpts) # (seq_len, feature)
    embs = self.integrate_sentences_info(embs) # (seq_len, hidden_size * 2)
    emb = embs[pos] # (hidden_size * 2)
    emb = emb.view(1, self.hidden_size * 2)
    o = self.classifier(emb) # (1, 2)
    loss = self.CEL(o, label)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = inpts.cuda()
      label = label.cuda()
    embs = self.cls(inpts) # (seq_len, feature)
    embs = self.integrate_sentences_info(embs) # (seq_len, hidden_size * 2)
    emb = embs[pos] # (hidden_size * 2)
    emb = emb.view(1, self.hidden_size * 2)
    o = self.classifier(emb) # (1, 2)
    self.print_train_info(o, label, -1)
    return o.argmax(1)

# ==============

G = {}

def init_G(length):
  G['ld'] = ld(ss_len = 2, max_len = 64)
  G['testld'] = tld(ss_len = 2, max_len = 64)
  G['devld'] = dld(ss_len = 2, max_len = 64)

def read_G():
  return G['m'], G['ld'], G['testld'], G['devld']


def get_datas(index, epoch, desc):
  m, ld , testld, devld = read_G()
  losses = R.get_datas(m, ld, testld, devld, index, epoch, desc)

def run():
  init_G(4)
  G['m'] = m = WikiSector(hidden_size = 256)
  get_datas(0, 1, 'dd')
  print(R.G)
  init_G(6)
  G['m'] = m = WikiSector(hidden_size = 256)
  get_datas(0, 1, 'dd')
  print(R.G)
