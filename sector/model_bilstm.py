import torch as t
import torch.nn as nn
import torch.nn.functional as F
import data_operator as data
from itertools import chain
import torch.optim as optim
import time
import random
import utils as U

# logger
import logging


# ids = [2, 5]
# length = 6
def ids2labels(ids, length):
  ids = ids.copy()
  result = []
  start = 0
  ids.append(length)
  for end in ids:
    squre_row = [1 if i in range(start, end) else 0 for i in range(0, length)]
    for i in range(start, end):
      result.append(squre_row)
    start = end
  return result
  
  

def beutiful_print(mat):
  for row in mat:
    should_print = []
    for item in row: 
      should_print.append(round(item.item(), 2))
    print(should_print)


class Model_BiLSTM(nn.Module):

  def SGD_train(self, epochs = 5, datas = data.read_data()):
    length = len(datas)
    start = time.time()
    for e in range(epochs):
      logging.info(f'start epoch{e}')
      shuffled = datas.copy()
      random.shuffle(shuffled)
      counter = 0
      for temp in shuffled:
        inpt = temp[0]
        label = temp[1]
        counter += 1
        logging.debug(f'training: {counter}/{length}')
        _,_ = self.train(inpt, label)
    end = time.time()
    logging.info(f'Trained! Epochs: {epochs}, dataset length: {len(datas)}, Time cost: {end - start} seconds')
    

  def init_hook(self):
    self.BCE = t.nn.BCELoss(None, None, None, 'sum')

  def init_embedding_layer(self):
    self.minify_layer = t.nn.Linear(self.s_bert_out_size, self.input_size)

  def init_logger(self):
    print('inited logger!!!')
    logging.basicConfig(
      filename='model_bilstm.log',
      format='%(asctime)s %(levelname)-8s %(message)s',
      level=logging.DEBUG,
      datefmt='%Y-%m-%d %H:%M:%S')
    

  def __init__(self,  input_size = 50, hidden_state_size = 50, verbose = False):
    super().__init__()
    hidden_state_size = input_size
    self.verbose = verbose
    self._define_variables(hidden_state_size, input_size)
    self.init_embedding_layer()
    self.encoder = t.nn.LSTM(self.input_size, self.hidden_size, 1, True, False, 0, True) # Bi-Direction
    self.sigmoid = t.nn.Sigmoid()
    self.init_hook()
    self.init_optim()
    self.init_logger()

  # after_softmax: (n)
  # labels: (n); item = 0 | 1
  def multiple_pointer_loss(self, scores, labels):
    return self.BCE(self.sigmoid(scores), labels)

  def get_should_update(self):
    return chain(self.encoder.parameters(), self.minify_layer.parameters())

  def init_optim(self):
    self.optim = optim.SGD(self.get_should_update(), lr=0.01, momentum=0.9)

  def _define_variables(self, hidden_state_size, input_size):
    self.input_size = hidden_state_size
    self.hidden_size = hidden_state_size
    # 2000以上的数字用于特殊用途
    # others
    self.MAX_INT = 999999
    # for sentences
    self.s_bert_out_size = 768
    self.batch_size = 1

  def print_info_this_step(self, inpt, target, loss):
    if self.verbose:
      print('What I want:')
      beutiful_print(target)
      print('What I got:')
      beutiful_print(inpt)
      print(f'The loss: {loss.item()}')

  def get_loss_by_input_and_target(self, inpts, targets):
    return self.BCE(inpts, targets)

  def zero_diagonal(self, mat):
    zero_diagonal = ones_yet_zeros_diagonal(len(mat))
    return mat * zero_diagonal

  def get_scores(self, outs, embs):
    # zero_diagonal = ones_yet_zeros_diagonal(len(labels))
    return self.zero_diagonal(self.get_scores_old(outs))

  def get_embs_from_inpts(self, inpts):
    return self.minify_layer(t.stack([data.sentence_to_embedding(s) for s in inpts]))

  def labels_processed(self, labels, inpts):
    return self.zero_diagonal(t.FloatTensor(ids2labels(labels, len(inpts))).detach()) # No grad

  def train(self, inpts, labels):
    if len(inpts) < 1:
      print('Warning: empty training sentence list')
      return
    embs = self.get_embs_from_inpts(inpts)
    embs = embs.view(-1, 1, self.input_size) # (?, 1, input_size)
    labels = self.labels_processed(labels, inpts) # (seq_len, seq_len)
    beutiful_print(labels)
    outs, (_, _) = self.encoder(embs) # (seq_len, 1, input_size * 2)

    scores = self.get_scores(outs, embs)

    loss = self.get_loss_by_input_and_target(scores, labels)
    self.print_info_this_step(scores, labels, loss)
    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    return scores.tolist(), labels.tolist()

  def get_scores_old(self, outs):
    temp_outs = outs.view(-1, self.input_size * 2)
    scores = self.sigmoid(t.mm(temp_outs, temp_outs.T)) # (seq_len, seq_len)
    return scores

  def dry_run(self, inpts):
    embs = self.get_embs_from_inpts(inpts)
    embs = embs.view(-1, 1, self.input_size)
    outs, (_, _) = self.encoder(embs) # (seq_len, 1, input_size * 2)
    # f = dot
    scores = self.get_scores(outs, embs)
    return scores.tolist()

  def output(self, mat, ss, ids, path='dd.png'):
    ss = [s[0:5] for s in ss]
    for i in ids:
      ss[i] = '$ ' + ss[i]
    U.output_heatmap(mat, ss, ss, path)
    
  def dry_run_then_output(self, datas, index):
    mat = self.dry_run(datas[index][0])
    self.output(mat, datas[index][0], datas[index][1])



def ones_yet_zeros_diagonal(size):
  zero_diagonal = t.ones(size, size)
  for i in range(size):
    zero_diagonal[i][i] = 0
  return zero_diagonal




class Model_MSE(Model_BiLSTM):
  def init_hook(self):
    print('init_hook(): do nothing here')

  def get_loss_by_input_and_target(inpts, targets):
    return (targets - inpts).pow(2).sum() #  MSE Loss as the same


class Model_BCE_with_buff_layer(Model_BiLSTM):
  def get_should_update(self):
    return chain(self.encoder.parameters(), self.minify_layer.parameters(), self.Q.parameters(), self.T.parameters())

  def init_hook(self):
    # self.BCE = nn.BCELoss(None, None, None, 'sum')
    self.BCE = nn.BCELoss(None, None, None, 'mean')
    self.Q = nn.Linear(self.input_size * 2, self.input_size * 2) # query buffer layer
    self.T = nn.Linear(self.input_size * 2, self.input_size * 2) # target buffer layer

  def get_scores(self, outs, embs):
    # labels = self.zero_diagonal(labels)
    querys = self.Q(outs).view(-1, self.input_size * 2)
    targets = self.T(outs).view(-1, self.input_size * 2)
    scores = self.zero_diagonal(self.sigmoid(t.mm(querys, targets.T))) # (seq_len, seq_len)
    # loss = self.get_loss_by_input_and_target(scores, labels)
    return scores


class Model_MSE_Adam(Model_MSE):
  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update())

class Model_BCE_Adam(Model_BiLSTM):
  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update())

class Model_BCE_Adam_Keep_Diagonal(Model_BiLSTM):

  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update())

  def zero_diagonal(self, mat):
    return mat


class Model_BCE_SGD0001(Model_BiLSTM):
  def init_optim(self):
    self.optim = optim.SGD(self.get_should_update(), lr=0.001, momentum=0.9)

