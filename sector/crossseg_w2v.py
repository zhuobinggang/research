import word2vec_fucker as w2v
import data_jap_reader as data
import torch as t
import torch.optim as optim
from itertools import chain
import danraku_runner_simple as runner
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
nn = t.nn

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
  lines = read_lines(data_id)
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

class MyDataset(Dataset):
  def __init__(self, half_window_size = 1):
    super().__init__()
    self.feature_size = 300
    self.init_datas_hook()
    self.half_window_size = half_window_size

  def __getitem_org__(self, idx):
    if idx >= len(self.datas) - 1:
      print(f'Warning: Should not get idx={idx}')
      return None
    left = []
    start = max(0, idx + 1 - self.half_window_size)
    end = min(idx + 1, len(self.datas))
    for i in range(start, end):
      left.append(self.datas[i])
    right = []
    start = max(0, idx + 1)
    end = min(idx + 1 + self.half_window_size, len(self.datas))
    for i in range(start, end):
      right.append(self.datas[i])
    label = 1 if right[0].startswith('\u3000') else 0
    left = no_indicator(left)
    right = no_indicator(right)
    return (left,right), label


  # return: (left: (128, 300), right: (128, 300)), label
  # pad with zero if not enough 
  def __getitem__(self, idx, tokens = 128):
    lefts = []
    rights = []
    left_stop = False
    right_stop = False
    # Get enough tokens
    for i in range(1, 10):
      self.half_window_size = i
      (left,right), label = self.__getitem_org__(idx)
      if not left_stop:
        new_lefts = w2v.sentence_to_wordvecs('。'.join(left))
        if len(lefts) == len(new_lefts) or len(new_lefts) > tokens:
          # print('。'.join(left) + '\n\n')
          left_stop = True
        lefts = new_lefts
      if not right_stop:
        new_rights = w2v.sentence_to_wordvecs('。'.join(right))
        if len(rights) == len(new_rights) or len(new_rights) > tokens:
          # print('。'.join(right)  + '\n\n')
          right_stop = True
        rights = new_rights
      if left_stop and right_stop:
        break
    # Trim
    lefts = t.tensor(lefts[-tokens:]) # (?, 300)
    rights = t.tensor(rights[0:tokens])
    # Pad as tensor
    try:
      if lefts.shape[0] < tokens:
        zeros = t.zeros(tokens - lefts.shape[0], self.feature_size)
        lefts = t.cat((zeros, lefts))
      if rights.shape[0] < tokens:
        zeros = t.zeros(tokens - rights.shape[0], self.feature_size)
        rights = t.cat((rights, zeros))
    except Exception as e:
      self.half_window_size = 1
      (left,right), label = self.__getitem_org__(idx)
      print('Exception occur!!!!')
      print(left)
      print(right)
      print(label)
    return (lefts,rights), label

  def init_datas_hook(self):
    self.datas = []

  def __len__(self):
    return len(self.datas) - 1

  def shuffle(self):
    random.shuffle(self.datas)


class Train_DS(MyDataset):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS(MyDataset):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS(MyDataset):
  def init_datas_hook(self):
    self.datas = data.read_devs()

class Loader():
  def __len__(self):
    length = len(self.ds.datas)
    divided = length / self.batch_size
    return int(divided) + 1 if int(divided) < divided else int(divided)

  def __init__(self, ds, batch_size = 4):
    self.start = 0
    self.ds = ds
    self.dataset = ds
    self.batch_size = batch_size

  def __iter__(self):
    return self

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start == len(self.ds):
      self.start = 0
      raise StopIteration()
    results = []
    end = min(self.start + self.batch_size, len(self.ds))
    for i in range(self.start, end):
      results.append(self.ds[i])
    self.start = end
    return (t.stack([d[0][0] for d in results]), t.stack([d[0][1] for d in results])), t.LongTensor([d[1] for d in results])

  def shuffle(self):
    self.ds.shuffle()

# ======

class Model(nn.Module):
  def __init__(self, hidden_size = 256):
    super().__init__()
    self.hidden_size = hidden_size  
    self.gru = nn.GRU(300, hidden_size, batch_first=True)
    self.classifier = nn.Sequential(
      nn.Linear(hidden_size * 2, hidden_size),
      nn.LeakyReLU(0.2),
      nn.Linear(hidden_size, 2),
    )
    self.optim = optim.AdamW(self.get_should_update(), 1e-3)
    self.CEL = nn.CrossEntropyLoss()
    self.verbose = False

  def set_verbose(self):
    self.verbose = not self.verbose

  def print_train_info(self, o, labels):
    if self.verbose:
      print(f'Want: {labels.tolist()} Got: {o.argmax(1).tolist()} ')

  def get_should_update(self):
    return chain(self.gru.parameters(), self.classifier.parameters())

  # (batch, token_size, 300)
  # return: (batch, hidden_size)
  def encode(self, tokens):
    out, hn = self.gru(tokens)
    return hn.view(tokens.shape[0], -1)

  # lefts = rights: (batch, token_size, 300)
  # labels : (batch)
  def train(self, inpts, labels):
    lefts, rights = inpts
    left = self.encode(lefts) # (batch, hidden_size)
    right = self.encode(rights) # (batch, hidden_size)
    encoded = t.stack([t.cat((l, r)) for l,r in zip(left, right)]) # (batch, hidden_size * 2)
    o = self.classifier(encoded) # (batch, 2)
    loss = self.CEL(o, labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels)
    return loss.detach().item()

  def dry_run(self, inpts):
    lefts, rights = inpts
    left = self.encode(lefts) # (batch, hidden_size)
    right = self.encode(rights) # (batch, hidden_size)
    encoded = t.stack([t.cat((l, r)) for l,r in zip(left, right)]) # (batch, hidden_size * 2)
    o = self.classifier(encoded) # (batch, 2)
    self.print_train_info(o, t.LongTensor([0]))
    return o.argmax(1).item() 
   

# =================

# ld = Loader(Train_DS(), 4)
# testld = Loader(Test_DS(), 1)

# ld = DataLoader(dataset=Train_DS(), batch_size=4, shuffle=True, num_workers=1)
# testld = DataLoader(dataset=Test_DS(), batch_size=1, shuffle=True, num_workers=1)
# 
# def set_test():
#   ld.dataset.datas = ld.dataset.datas[:100]
#   testld.dataset.datas = testld.dataset.datas[:100]
# 
# set_test()
