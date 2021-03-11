import random
import torch as t
import numpy as np

def generate_data(num):
  input_range = [1, 19]
  seq_size_range = [3, 5]
  result = []
  for i in range(num):
    temp = []
    for j in range(random.randint(seq_size_range[0], seq_size_range[1])):
      new_num = random.randint(input_range[0], input_range[1])
      while new_num in temp:
        new_num = random.randint(input_range[0], input_range[1])
      temp.append(new_num)
    result.append(temp)
  return result

def regenerate_train_and_test(train_lines = 900, test_lines = 100):
  train_data = generate_data(train_lines)
  test_data = generate_data(test_lines)
  with open ('train.txt', 'w') as f: 
    for data in train_data:
      f.write(str(data).replace('[','').replace(']','') + '\n')
  with open ('test.txt', 'w') as f: 
    for data in test_data:
      f.write(str(data).replace('[','').replace(']','') + '\n')

def read_data(filename='train.txt'):
  result = []
  with open (filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
       result.append(list(map(lambda x: int(x), line.rstrip().split(','))))
  return result

def read_train():
  return read_data('train.no_repeat_3000.txt')


def read_test():
  return read_data('test.no_repeat_200.txt')

def ordered_index(list_of_num, MAX_INT = 99999):
  l = list_of_num.copy()
  result = []
  minus = MAX_INT
  record_index = -1
  for _ in range(len(l)):
    minus = MAX_INT
    record_index = -1
    for index,num in enumerate(l):
      if num != MAX_INT and num < minus:
        minus = num
        record_index = index
    if record_index != -1:
      l[record_index] = MAX_INT
      result.append(record_index)
  return result

class Dataset(t.utils.data.dataset.Dataset):
  def __init__(self, datas = []):
    super().__init__()
    self.datas = datas
    self.init_hook()

  def init_hook(self):
    pass

  def __getitem__(self, start):
    return self.datas[start]

  def __len__(self):
    return len(self.datas)

  def shuffle(self):
    random.shuffle(self.datas)

class Loader():
  def __init__(self, ds, batch_size = 12):
    self.ds = ds
    self.dataset = ds
    self.start = 0
    self.batch_size = batch_size

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.ds.datas)

  def processed_data(self, start):
    nums = self.ds[start] # list of number
    inpts = t.LongTensor(nums)
    indexs =  ordered_index(nums) # (lenth_of_nums + 1, lenth_of_nums + 1)
    indexs += [len(indexs)] # 增加EOS
    # labels = t.tensor(np.eye(len(indexs), dtype=np.long)[indexs])
    labels = t.LongTensor(indexs)
    return inpts, labels

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start >= len(self):
      self.start = 0
      raise StopIteration()
    else:
      self.start += 1
      return self.processed_data(self.start - 1)

  def shuffle(self):
    self.ds.shuffle()
