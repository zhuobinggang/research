import numpy as np
from transformers import BertModel, BertJapaneseTokenizer
import data_jap_reader as data
import torch as t
from importlib import reload
import danraku_runner_simple as runner
import torch.optim as optim
from itertools import chain
import logging
nn = t.nn

GPU_OK = t.cuda.is_available()


class DatasetAbstract(t.utils.data.dataset.Dataset):
  def __init__(self, half_window_size = 1):
    super().__init__()
    self.feature_size = 300
    self.half_window_size = half_window_size
    self.init_hook()
    self.init_datas_hook()

  def init_hook(self):
    pass

  def no_indicator(self, ss):
    return [s.replace('\u3000', '') for s in ss]

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
    left = self.no_indicator(left)
    right = self.no_indicator(right)
    return (left,right), label


  # return: (left: (128, 300), right: (128, 300)), label
  # pad with zero if not enough 
  def __getitem__(self, idx, tokens = 128):
    pass

  def init_datas_hook(self):
    self.datas = []

  def __len__(self):
    return len(self.datas) - 1

  def shuffle(self):
    random.shuffle(self.datas)

# ==============


# Trim length with max = 128
class Dataset(DatasetAbstract):
  def token_encode(self, text):
    ids = self.toker.encode(text, add_special_tokens = False)
    if self.max_size < len(ids):
      logging.warning(f'Length {len(ids)} exceed!: {text[:30]}...')
    ids = ids[0: self.max_size]
    return ids

  def init_toker(self):
    toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.toker = toker
    self.pad_id = toker.pad_token_id
    self.cls_id = toker.cls_token_id
    self.sep_id = toker.sep_token_id
    self.sentence_period = '。'

  def init_hook(self):
    self.tokens = 128
    self.max_size = 128
    self.init_toker()

  def joined(self, ss):
    return self.sentence_period.join(ss)

  def trim_ids(self, lefts, rights):
    tokens = self.tokens
    lefts = lefts[-tokens:] # (<=tokens), list
    rights = rights[0:tokens] # (<=tokens), list
    return lefts, rights

  def pad_and_create_attend_mark_with_special_token(self, lefts, rights):
    tokens = self.tokens
    attend_left = []
    attend_right = []
    if len(lefts) < tokens:
      attend_left = np.repeat(0, tokens - len(lefts)).tolist() + np.repeat(1, len(lefts)).tolist()
      topad = np.repeat(self.pad_id, tokens - len(lefts)).tolist()
      lefts = topad + lefts # (=tokens), list
    else:
      attend_left = np.repeat(1, tokens).tolist()
    if len(rights) < tokens:
      attend_right = np.repeat(1, len(rights)).tolist() + np.repeat(0, tokens - len(rights)).tolist()
      topad = np.repeat(self.pad_id, tokens - len(rights)).tolist()
      rights = rights + topad
    else:
      attend_right = np.repeat(1, tokens).tolist()
    # Add special token
    results_ids = [self.cls_id] + lefts + [self.sep_id] + rights
    attend_left = [1] + attend_left
    attend_right = [1] + attend_right
    attend_mark = attend_left + attend_right
    return results_ids, attend_mark
  
  # return: (left: (128, 300), right: (128, 300)), label
  # pad with zero if not enough 
  def __getitem__(self, idx):
    tokens = self.tokens
    lefts = []
    rights = []
    left_stop = False
    right_stop = False
    # Get enough tokens
    for i in range(1, 10):
      self.half_window_size = i
      (left,right), label = self.__getitem_org__(idx)
      if not left_stop:
        left_token_ids = self.token_encode(self.joined(left))
        if len(lefts) == len(left_token_ids) or len(left_token_ids) > tokens:
          # print('。'.join(left) + '\n\n')
          left_stop = True
        lefts = left_token_ids
      if not right_stop:
        right_token_ids = self.token_encode(self.joined(right))
        if len(rights) == len(right_token_ids) or len(right_token_ids) > tokens:
          # print('。'.join(right)  + '\n\n')
          right_stop = True
        rights = right_token_ids
      if left_stop and right_stop:
        break
    # Trim
    lefts, rights = self.trim_ids(lefts, rights)
    # Pad and create attend mark and add special token
    results_ids, attend_mark = self.pad_and_create_attend_mark_with_special_token(lefts, rights)
    return (results_ids, attend_mark), label

class Dataset_Without_Trim_Length(Dataset):
  def token_encode(self, text):
    ids = self.toker.encode(text, add_special_tokens = False)
    return ids

class Train_DS(Dataset_Without_Trim_Length):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS(Dataset_Without_Trim_Length):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS(Dataset_Without_Trim_Length):
  def init_datas_hook(self):
    self.datas = data.read_devs()
  

# ==========

class LoaderAbstract():
  def __len__(self):
    length = len(self.ds)
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

class Loader(LoaderAbstract):
  def __init__(self, ds, batch_size = 4):
    super().__init__(ds, batch_size)
    self.init_hook()

  def init_hook(self):
    pass

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
    return (t.LongTensor([d[0][0] for d in results]), t.LongTensor([d[0][1] for d in results])), t.LongTensor([d[1] for d in results])

# ==========

class Model(nn.Module):
  def __init__(self, weight_one = 1, hidden_size = 256):
    super().__init__()
    self.max_memory_batch = 6
    self.hidden_size = hidden_size
    self.bert_size = 768
    self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, weight_one]))
    self.verbose = False
    self.init_bert()
    self.init_hook()
    self.optim = optim.AdamW(self.get_should_update(), self.learning_rate())
    print('Init AdamW with lr = {self.learning_rate()}')
    if GPU_OK:
      _ = self.cuda()

  def learning_rate(self):
    return 2e-5

  def init_hook(self):
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, int(self.bert_size / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.bert_size / 2), 2),
    )

  def init_bert(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()

  def set_verbose(self):
    self.verbose = not self.verbose

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {o.argmax(1).tolist()} Loss: {loss} ')

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters())

  # token_ids = attend_marks: (batch, seq_len)
  # return: (batch, 768)
  def get_batch_cls_emb(self, token_ids, attend_marks):
    if token_ids.shape[0] <= self.max_memory_batch:
      result = self.bert(token_ids, attend_marks, return_dict=True)
      return result.pooler_output
    else:
      batch_token_ids = token_ids.split(self.max_memory_batch)
      batch_attend_marks = attend_marks.split(self.max_memory_batch)
      batch_results = [self.bert(token_ids, attend_marks, return_dict=True) for token_ids, attend_marks in zip(batch_token_ids, batch_attend_marks)]
      batch_results = [res.pooler_output for res in batch_results] #(?, mini_batch, 768)
      return t.cat(batch_results)

  def processed_embs(self, embs):
    return embs

  # inpts: (token_ids, attend_marks), list
  # token_ids = attend_marks: (batch, seq_len), LongTensor
  # labels: (batch), LongTensor
  def train(self, inpts, labels):
    token_ids, attend_marks = inpts
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      labels = labels.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (batch, 768)
    embs = self.processed_embs(embs)
    o = self.classifier(embs) # (batch, 2)
    loss = self.CEL(o, labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    token_ids, attend_marks = inpts
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (batch, 768)
    o = self.classifier(embs) # (batch, 2)
    self.print_train_info(o, labels, -1)
    return o.argmax(1)
