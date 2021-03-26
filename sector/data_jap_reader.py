import random
import numpy as np

def read_docs(data_id = 1):
  if data_id == 1:
    filenames = ['sansirou', 'sorekara', 'mon', 'higan', 'gyoujin'] 
  elif data_id == 2:
    filenames = ['kokoro'] 
  elif data_id == 3:
    filenames = ['meian'] 
  paths = [f'datasets/{name}_new.txt' for name in filenames]
  docs = []
  for path in paths:
    with open(path) as f:
      lines = f.readlines()
      docs.append(lines)
  return docs

def read_lines(data_id = 1):
  docs = read_docs(data_id)
  lines = []
  for doc in docs:
    lines += doc
  return [line.replace(' ', '').replace('\n', '') for line in lines]

def read_sentences(data_id = 1): 
  lines = read_lines(data_id) # 隐式根据换行划分句子
  sentences = []
  for line in lines:
    ss = line.split('。')
    for s in ss:
      if len(s) > 1:
        sentences.append(s)
  return sentences

def read_trains():
  return read_sentences(1)

def read_tests():
  return read_sentences(3)

def read_devs():
  return read_sentences(2)

def no_indicator(ss):
  return [s.replace('\u3000', '') for s in ss]


class Dataset():
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
    start = max(start, 0)
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


def train_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len, max_ids = max_ids)
  ds.set_datas(read_trains())
  return ds

def test_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len, max_ids = max_ids)
  ds.set_datas(read_tests())
  return ds

def dev_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len, max_ids = max_ids)
  ds.set_datas(read_devs())
  return ds
