import torch as t
import torch.nn as nn
import torch.nn.functional as F
import data_operator as data
from itertools import chain
import torch.optim as optim


# ids = [2, 5]
# length = 6
def ids2labels(ids, length):
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

  def __init__(self,  input_size = 50, hidden_state_size = 50, verbose = False):
    super().__init__()
    self.verbose = verbose
    self._define_variables(hidden_state_size, input_size)
    self.minify_layer = t.nn.Linear(self.s_bert_out_size, self.input_size)
    self.encoder = t.nn.LSTM(self.input_size, self.hidden_size, 1, True, False, 0, True) # Bi-Direction
    self.sigmoid = t.nn.Sigmoid()
    self.BCE = t.nn.BCELoss(None, None, None, 'sum')
    self.init_optim()

  # after_softmax: (n)
  # labels: (n); item = 0 | 1
  def multiple_pointer_loss(self, scores, labels):
    return self.BCE(self.sigmoid(scores), labels)

  def init_optim(self):
    should_update = chain(self.encoder.parameters(), self.minify_layer.parameters())
    self.optim = optim.SGD(should_update, lr=0.01, momentum=0.9)

  def _define_variables(self, hidden_state_size, input_size):
    self.input_size = hidden_state_size
    self.hidden_size = hidden_state_size
    # 2000以上的数字用于特殊用途
    # others
    self.MAX_INT = 999999
    # for sentences
    self.s_bert_out_size = 768
    self.batch_size = 1

  def train(self, ss, ids):
    if len(ss) < 1:
      print('Warning: empty training sentence list')
      return
    embs = self.minify_layer(t.stack([data.sentence_to_embedding(s) for s in ss]))
    embs = embs.view(-1, 1, self.input_size)
    # print(embs.shape)
    outs, (_, _) = self.encoder(embs)
    # print(outs.shape)
    temp_outs = outs.view(-1, self.input_size * 2)
    # print(temp_outs.shape)
    # f = dot
    scores = self.sigmoid(t.mm(temp_outs, temp_outs.T)) # (seq_len, seq_len)
    labels = t.FloatTensor(ids2labels(ids, len(ss)))
    loss = self.BCE(scores, labels)

    if self.verbose:
      print('What I want:')
      beutiful_print(labels)
      print('What I got:')
      beutiful_print(scores)
      print(f'The loss: {loss.item()}')

    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()


