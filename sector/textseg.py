import torch.nn.functional as F
import torch.optim as optim
import time
import random
import utils as U
from torch.nn.functional import pad

import model_bilstm as model
from torch.nn.utils.rnn import pad_sequence
import data_operator as data
import torch as t
import torch.nn as nn
from itertools import chain


def mark_id_as_one(ids, length):
  result = [0] * length
  for index in ids:
    result[index] = 1
  return result

# =================

class Model(nn.Module):

  def _define_variables(self, hidden_state_size, input_size):
    self.input_size = hidden_state_size
    self.hidden_size = hidden_state_size
    # others
    self.MAX_INT = 999999
    # for sentences
    self.s_bert_out_size = 768

  def get_should_update(self):
    return chain(self.solver.parameters(), self.minify.parameters(), self.classify.parameters())

  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update(), 0.01)

  def __init__(self,  input_size = 50, hidden_state_size = 50, verbose = False):
    super().__init__()
    self.verbose = verbose
    self._define_variables(hidden_state_size, input_size)
    self.solver = t.nn.LSTM(self.input_size, self.hidden_size, 2, True, False, 0, True)
    self.minify = t.nn.Linear(self.s_bert_out_size, self.input_size)
    self.classify = t.nn.Linear(self.input_size * 2, 2)
    self.softmax = t.nn.CrossEntropyLoss()
    # self.sigmoid = t.nn.Sigmoid()
    # self.BCE = t.nn.BCELoss(None, None, None, 'mean')
    self.init_optim()

  def get_embs_from_inpts(self, inpts):
    return self.minify(pad_sequence([data.ss_to_embs(ss) for ss in inpts])) # (max_len, batch_size, input_size)

  # labels: [[id]]
  # embs: (max_len, batch_size, input_size)
  # return: (max_len, batch_size, 1)
  def labels_processed(self, labels, embs):
    max_len, batch_size, _ = embs.shape
    results = [mark_id_as_one(ids, max_len) for ids in labels] # (batch_size, max_len)
    results = t.LongTensor(results).transpose(0, 1)
    # return 
    return results

  def print_info_this_step(self, inpt, target, loss):
    if self.verbose:
      #print('What I want:')
      #beutiful_print(target)
      #print('What I got:')
      #beutiful_print(inpt)
      print(f'The loss: {loss.item()}')
  
  # inpts: [[sentence]]
  # labels: [[id]]
  def train(self, inpts, labels):
    if len(inpts) < 1:
      print('Warning: empty training sentence list')
      return
    embs = self.get_embs_from_inpts(inpts) # (max_len, batch_size, input_size)
    outs, (_, _) = self.solver(embs) # (max_len, batch_size, input_size * 2)
    scores = self.classify(outs) # (max_len, batch_size, 2)

    labels_processed = self.labels_processed(labels, embs) # (max_seq_len, batch_size) & type = Long
    loss = self.softmax(scores.transpose(1, 2), labels_processed)

    self.print_info_this_step(scores, labels_processed, loss)
    
    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    return scores.tolist(), labels_processed.tolist()

  @t.no_grad()
  def dry_run(self, inpts):
    if len(inpts) > 1:
      print('No, I do not run for batch!')
      return None
    embs = self.get_embs_from_inpts(inpts) # (max_len, batch_size, input_size)
    outs, (_, _) = self.solver(embs) # (max_len, batch_size, input_size * 2)
    scores = self.classify(outs) # (max_len, batch_size, 2)
    return scores.tolist()

  def cal_Pk_by_loader(self, loader):
    return cal_Pk(self, loader)

# =================

def dry_run_get_ids(m, loader):
  loader.batch_size = 1
  loader.start = 0
  ids = []
  for inpts, _ in loader:
    one_or_zeros = t.argmax(t.tensor(m.dry_run(inpts)), 2).view(-1) # (seq_len)
    ids.append([index for index, one_or_zero in enumerate(one_or_zeros) if one_or_zero == 1])
  return ids

def cal_Pk(m, loader):
  ids = dry_run_get_ids(m, loader)
  return U.cal_Pk(loader.ds.datas, ids)
