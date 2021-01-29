import re
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from torch.nn.utils.rnn import pad_sequence
from itertools import chain
import torch.nn as nn
import torch.optim as optim
import logging
import time
import random
# from torch.utils.data import Dataset
t = torch
tokenizer = None
model = None


def init_logger(path):
  logging.basicConfig(
    filename=path,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


class Dataset:
  def __init__(self):
    self.init_datas_hook()

  def init_datas_hook(self):
    self.datas = read_data()

  def __len__(self):
    return len(self.datas)

  def __getitem__(self, idx):
    return self.datas[idx][1], int(self.datas[idx][0])

  def shuffle(self):
    random.shuffle(self.datas)

class Loader:
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

def read_data():
  f = open('train.csv')
  lines = f.readlines()
  f.close()
  lines = lines[1:] # remove header
  results = [(label, text) for (index, label, text) in [re.match('^([0-9]+)\,([0-9])\,(.*)', line).groups() for line in lines]]
  return results

def try_init_bert():
  global tokenizer, model
  if tokenizer == None:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  if model == None:
    model = BertModel.from_pretrained('bert-base-uncased')

# texts: (batch_size, string)
def texts2embs(texts):
  try_init_bert()
  batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
  input_ids = batch['input_ids']
  attention_mask = batch['attention_mask']
  with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(input_ids, attention_mask)
  return outputs[0] # (batch_size, max_len, 768)
  
class Model_LSTM(nn.Module):
  def init_encoder():
    self.encoder = t.nn.LSTM(self.input_size, self.hidden_size) # Bi-Direction

  def __init__(self):
    super().__init__()
    self.input_size = 50
    self.hidden_size = 50
    self.minify = t.nn.Linear(768, self.input_size)
    self.init_encoder() # Bi-Direction
    self.classify = t.nn.Linear(self.hidden_size, 2)
    self.softmax = t.nn.CrossEntropyLoss()
    self.init_optim()
    self.verbose = False

  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update(), 0.01)

  def get_should_update(self):
    return chain(self.minify.parameters(), self.encoder.parameters(), self.classify.parameters())

  # inpts: (batch_size, ?)
  def get_embs_from_inpts(self, inpts):
    return self.minify(texts2embs(inpts)).transpose(0, 1) # (max_len, batch_size, input_size)

  # labels: (batch_size, 0/1)
  # return: (1, batch_size)
  def labels_processed(self, labels, _):
    result = t.LongTensor(labels) # (batch_size)
    return result.view(1, -1) # (1, batch_size)


  def print_info_this_step(self, inpt, target, loss):
    if self.verbose:
      print(f'The loss: {loss.item()}')

  def get_outs_by_embs(self, embs):
    _, (out, _) = self.encoder(embs) # (1, batch_size, input_size)
    return out

  # inpts: (batch_size, ?)
  # labels: (batch_size, 0/1)
  def train(self, inpts, labels):
    embs = self.get_embs_from_inpts(inpts) # (max_len, batch_size, input_size)
    out = self.get_outs_by_embs(embs) # (1, batch_size, hidden_size)
    scores = self.classify(out) # (1, batch_size, 2)
    labels_processed = self.labels_processed(labels, embs) # (1, batch_size)
    loss = self.softmax(scores.transpose(1, 2), labels_processed)
    self.print_info_this_step(scores, labels_processed, loss)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

  # inpts: (1, ?)
  # return: (2)
  def dry_run(self, inpts):
    embs = self.get_embs_from_inpts(inpts) # (max_len, 1, input_size)
    out = self.get_outs_by_embs(embs) # (1, 1, input_size)
    scores = self.classify(out) # (1, 1, 2)
    return scores.view(2)

    
class Model_TF(Model_LSTM):
  def init_encoder(self):
    self.encoder = nn.TransformerEncoderLayer(50, 1, 50, 0) # Bi-Direction

  def get_outs_by_embs(self, embs):
    outs = self.encoder(embs) # (max_len, batch_size, hidden_size)
    _, batch_size, hidden_size = outs.shape
    return outs.mean(0).view(1, batch_size, hidden_size) # (1, batch_size, hidden_size)


def train_by_data_loader(m, loader, epoch = 5, logger = print):
  length = len(loader.ds)
  start = time.time()
  for e in range(epoch):
    logger(f'start epoch{e}')
    loader.shuffle()
    for inpts, labels in loader:
      logger(f'{loader.start}/{length}')
      m.train(inpts, labels)
  end = time.time()
  logger(f'Trained! Epochs: {epoch}, Batch size: {loader.batch_size}, dataset length: {length}, Time cost: {end - start} seconds')

def test_by_loader(m, loader):
  loader.start = 0
  loader.batch_size = 1
  try_time = 0
  acc_time = 0
  fail_time = 0
  for inpts, labels in loader:
    out = dry_run(m, inpts).argmax().item()
    try_time += 1
    if out != labels[0]:
      fail_time += 1
    else:
      acc_time += 1
  return (try_time, acc_time, fail_time)
    

def dry_run(m, inpts):
  embs = m.get_embs_from_inpts(inpts) # (max_len, 1, input_size)
  out = m.get_outs_by_embs(embs) # (1, 1, input_size)
  scores = m.classify(out) # (1, 1, 2)
  return scores.view(2)

