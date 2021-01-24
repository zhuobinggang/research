import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import data_operator_hello_world as data_hello
import data_operator as data
import random

class MyDataset(Dataset):
  def __init__(self):
    self.init_datas_hook()

  def init_datas_hook(self):
    self.datas = []

  def __len__(self):
    return len(self.datas)

  def __getitem__(self, idx):
    return self.datas[idx][0], self.datas[idx][1]

  def shuffle(self):
    random.shuffle(self.datas)


class HelloDataset(MyDataset):
  def init_datas_hook(self):
    self.datas = data_hello.read()
    

class WikiSectionDataset(MyDataset):
  def __init__(self, test=False):
    if not test:
      self.datas = data.read_trains()
    elif test:
      self.datas = data.read_tests()
      
# =================

class Loader():
  def __init__(self, ds, batch_size = 4):
    self.start = 0
    self.ds = ds
    self.batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self):
    if self.start == len(self.ds):
      self.start = 0
      raise StopIteration()
    results = []
    end = min(self.start + self.batch_size, len(self.ds))
    for i in range(self.start, end):
      results.append(self.ds[i])
    self.start = end
    return [d[0] for d in results], [d[1] for d in results]

  def shuffle(self):
    self.ds.shuffle()

# =================
    
def example():
  loader = HelloLoader()
  for inpts, labels in loader:
    m.train(inpts, labels)
  loader.shuffle()


