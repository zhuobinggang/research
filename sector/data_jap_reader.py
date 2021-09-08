import random
import numpy as np
import re

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
    if len(ss) == 1: # 如果句中没有句号，说明是根据换行分的，大概是对话
      sentences.append(ss[0])
    else:
      for i,s in enumerate(ss):
        if len(s) > 1: # 恢复句号，因为句号重要，但是最后一句不需要，也需要提防空白的情况
          if i != len(ss) - 1: 
            sentences.append(s + '。')
          else:
            sentences.append(s) # 最后一句不需要加句号
  return sentences

# 废弃： 这个方法行不通，一个对话框可以跨越很多行
def read_sentences_v2(data_id = 1): 
  lines = read_lines(data_id) # 隐式根据换行划分句子
  mess = ''.join(lines)
  mess = re.sub(r'［.*?］', '', mess) # 去掉诸如此类强调：［＃「帰って来た」は底本では「帰った来た」］
  dd = re.findall(r'「.*?」|[^「^」]*?。', mess2) # length = 26942，不知为何少了很多, 原文27823
  sentences = []
  return sentences

def read_sentences_v3(data_id = 1): 
  lines = read_lines(data_id) # 隐式根据换行划分句子
  lines = replace_emphasize(lines) # 删除所有强调
  sentences = []
  for line in lines:
    ss = line.split('。')
    if len(ss) == 1: # 如果一行中没有句号，还蛮多这种情况的
      sentences.append(ss[0])
    else:
      for i,s in enumerate(ss):
        if len(s) > 1: # 恢复句号，因为句号重要，但是最后一句不需要，也需要提防空白的情况
          if i != len(ss) - 1: # 如果不是最后一句, 恢复句号
            sentences.append(s + '。')
          else: # 如果是最后一句，说明本身就没有句号
            if s == '……' or s == '――': # 排除奇怪结尾的情况
              pass
            else:
              sentences.append(s) # 最后一句不需要加句号
        else: # 如果为‘’，即空白，什么也不用做
            pass
  return sentences

def replace_emphasize(lines):
  lines = [line.replace('$', '') for line in lines]
  mess = '$'.join(lines)
  mess = re.sub(r'［.*?］', '', mess) # 去掉诸如此类强调：［＃「帰って来た」は底本では「帰った来た」］
  lines = mess.split('$')
  return lines

def read_sentences_v4(data_id = 1): 
  lines = read_lines(data_id) 
  lines = replace_emphasize(lines) # 删除所有强调符号
  mess = ''.join(lines)
  dd = re.findall(r'.*?[。」]', mess) # 所有以句号或者问号结尾的都当成句子
  return dd # length = 24866, 原文27823

def read_trains():
  return read_sentences(1)

def read_tests():
  return read_sentences(3)

def read_devs():
  return read_sentences(2)

def no_indicator(ss):
  return [s.replace('\u3000', '') for s in ss]


class Dataset():
  def __init__(self, ss_len = 8, datas=[]):
    super().__init__()
    self.ss_len = ss_len
    self.datas = datas
    self.init_datas_hook()
    self.init_hook()
    self.start = 0

  def init_hook(self):
    pass

  def init_datas_hook(self):
    pass

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
  ds = Dataset(ss_len = ss_len)
  ds.set_datas(read_trains())
  return ds

def test_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len)
  ds.set_datas(read_tests())
  return ds

def dev_dataset(ss_len, max_ids):
  ds = Dataset(ss_len = ss_len)
  ds.set_datas(read_devs())
  return ds

# start ======================= Loader Tested, No Touch =======================
class Loader():
  def __init__(self, ds, half, batch):
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2 + 1
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()

  def __iter__(self):
    return self

  def __len__(self):
    return self.end_point() - self.start_point() + 1

  def start_point(self):
    return 0

  def end_point(self):
    return len(self.ds.datas) - 1

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    start = idx - self.half # 可能是负数
    ss, labels = self.ds[start] # 会自动切掉负数的部分
    correct_start = max(start, 0)
    pos = idx - correct_start
    return ss, labels, pos # 只需要中间的label

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  # raise StopIteration()
  def __next__(self):
    start = self.start
    if start > self.end_point():
      self.start = self.start_point()
      raise StopIteration()
    else:
      results = []
      end = min(start + self.batch - 1, self.end_point())
      for i in range(start, end + 1):
        ss, label, pos = self.get_data_by_index(i)
        results.append((ss, label, pos))
      self.start = end + 1
      return results

  def shuffle(self):
    self.ds.shuffle()


class Loader_Symmetry(Loader):
  def __init__(self, ds, half, batch):
    # print(f'init Loader_Symmetry half={half}, batch={batch}')
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    start = idx - self.half # 可能是负数
    ss, labels = self.ds[start] # 会自动切掉负数的部分
    correct_start = max(start, 0)
    pos = idx - correct_start
    return ss, labels, pos # 只需要中间的label


class Loader_SGD():
  def __init__(self, ds, half, batch, shuffle = True):
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2 + 1
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()
    ld = Loader(ds, half, batch=1)
    self.masses = []
    for mass in ld:
      ss, labels, pos = mass[0]
      self.masses.append((ss, labels, pos))
    if shuffle:
      self.shuffle()
      print('Loader_SGD: Shuffled')
    else:
      print('Loader_SGD: No Shuffle')

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.masses)

  def start_point(self):
    return 0

  def end_point(self):
    return len(self.masses) - 1

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    return self.masses[idx]

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  # raise StopIteration()
  def __next__(self):
    start = self.start
    if start > self.end_point():
      self.start = self.start_point()
      raise StopIteration()
    else:
      results = []
      end = min(start + self.batch - 1, self.end_point())
      for i in range(start, end + 1):
        ss, label, pos = self.get_data_by_index(i)
        results.append((ss, label, pos))
      self.start = end + 1
      return results

  def shuffle(self):
    random.shuffle(self.masses)

class Loader_Symmetry_SGD(Loader_SGD):
  def __init__(self, ds, half, batch, shuffle = True):
    # print(f'init Loader_Symmetry_SGD half={half}, batch={batch}')
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()
    ld = Loader_Symmetry(ds, half, batch=1)
    self.masses = []
    for mass in ld:
      ss, labels, pos = mass[0]
      self.masses.append((ss, labels, pos))
    if shuffle:
      random.shuffle(self.masses)
      # print('Loader_Symmetry_SGD: Shuffled')
    else:
      # print('Loader_Symmetry_SGD: No Shuffle')
      pass
  
# end ======================= Loader Tested, No Touch =======================


