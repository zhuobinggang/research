import crossseg_w2v as W
import numpy as np
from transformers import BertModel, BertJapaneseTokenizer
import data_jap_reader as data
import torch as t
from importlib import reload
import danraku_runner_simple as runner
import torch.optim as optim
from itertools import chain
nn = t.nn
toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
pad_id = toker.pad_token_id
cls_id = toker.cls_token_id
sep_id = toker.sep_token_id

def token_encode(text):
  return toker.encode(text, add_special_tokens = False)

class Dataset(W.MyDataset):
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
        left_token_ids = token_encode('。'.join(left))
        if len(lefts) == len(left_token_ids) or len(left_token_ids) > tokens:
          # print('。'.join(left) + '\n\n')
          left_stop = True
        lefts = left_token_ids
      if not right_stop:
        right_token_ids = token_encode('。'.join(right))
        if len(rights) == len(right_token_ids) or len(right_token_ids) > tokens:
          # print('。'.join(right)  + '\n\n')
          right_stop = True
        rights = right_token_ids
      if left_stop and right_stop:
        break
    # Trim
    lefts = lefts[-tokens:] # (<=tokens), list
    rights = rights[0:tokens] # (<=tokens), list
    # Pad and create attend mark
    attend_left = []
    attend_right = []
    if len(lefts) < tokens:
      attend_left = np.repeat(0, tokens - len(lefts)).tolist() + np.repeat(1, len(lefts)).tolist()
      topad = np.repeat(pad_id, tokens - len(lefts)).tolist()
      lefts = topad + lefts # (=tokens), list
    else:
      attend_left = np.repeat(1, tokens).tolist()
    if len(rights) < tokens:
      attend_right = np.repeat(1, len(rights)).tolist() + np.repeat(0, tokens - len(rights)).tolist()
      topad = np.repeat(pad_id, tokens - len(rights)).tolist()
      rights = rights + topad
    else:
      attend_right = np.repeat(1, tokens).tolist()
    # Add special token
    results_ids = [cls_id] + lefts + [sep_id] + rights
    attend_left = [1] + attend_left
    attend_right = [1] + attend_right
    attend_mark = attend_left + attend_right
    return (results_ids, attend_mark), label

class Train_DS(Dataset):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS(Dataset):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS(Dataset):
  def init_datas_hook(self):
    self.datas = data.read_tests()
  

# ==========

class Loader(W.Loader):
  def __init__(self, ds, batch_size = 4):
    super().__init__(ds, batch_size)

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
  def __init__(self):
    super().__init__()
    self.bert_size = 768
    self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 3]))
    self.verbose = False
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, int(self.bert_size / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.bert_size / 2), 2),
    )
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.optim = optim.AdamW(self.get_should_update(), 2e-5)

  def set_verbose(self):
    self.verbose = not self.verbose

  def print_train_info(self, o, labels, loss):
    if self.verbose:
      print(f'Want: {labels.tolist()} Got: {o.argmax(1).tolist()} Loss: {loss} ')

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters())

  # token_ids = attend_marks: (batch, seq_len)
  # return: (batch, 768)
  def get_batch_cls_emb(self, token_ids, attend_marks):
    result = self.bert(token_ids, attend_marks, return_dict=True)
    return result.pooler_output

  # inpts: (token_ids, attend_marks), list
  # token_ids = attend_marks: (batch, seq_len), LongTensor
  # labels: (batch), LongTensor
  def train(self, inpts, labels):
    token_ids, attend_marks = inpts
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (batch, 768)
    o = self.classifier(embs) # (batch, 2)
    loss = self.CEL(o, labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

  def dry_run(self, inpts):
    token_ids, attend_marks = inpts
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (batch, 768)
    o = self.classifier(embs) # (batch, 2)
    # self.print_train_info(o, t.LongTensor([0]), 0)
    return o.argmax(1)

# ============

ld = Loader(Train_DS(), 24)
testld = Loader(Test_DS(), 24)
devld = Loader(Dev_DS(), 24)

def set_test():
  ld.dataset.datas = ld.dataset.datas[:100]
  testld.dataset.datas = testld.dataset.datas[:50]

# return: (m, (prec, rec, f1, bacc), losss)
def run_test(m):
  set_test()
  m.verbose = True
  return runner.run(m, ld, testld, 2, batch=24)

def run(m):
  return runner.run(m, ld, testld, 2, batch=24)

def run_at_night():
  m = Model()
  _, results, losss = run(m)
  t.save(m, 'save/csg_bert.tch')
