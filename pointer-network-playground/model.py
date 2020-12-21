import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain


def ordered_index(list_of_num, MAX_INT = 99999):
  l = list_of_num
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

class Model(nn.Module):

  def _define_variables(self):
    self.input_range = [0,2000] # min, max
    self.num_embeddings = 2020
    # embedding_dim = 9 # Number embedding dimension
    self.input_size = 9
    self.hidden_size = self.input_size
    # 2000以上的数字用于特殊用途
    self.SOF = 2001
    self.EOF = 2002
    self.encoder_c0 = 2003
    self.decoder_c0 = 2004
    # others
    self.MAX_INT = 999999

  def __init__(self):
    super().__init__()
    self._define_variables()
    self.embedding = nn.Embedding(self.num_embeddings, self.input_size)
    self.embedding.requires_grad_(False)#TODO: Do not add to optim
    self.encoder = t.nn.LSTM(self.input_size, self.hidden_size)
    self.decoder = t.nn.LSTM(self.input_size, self.hidden_size)
    self.CEL = nn.CrossEntropyLoss()
    self.query_from_dh_layer = nn.Linear(self.hidden_size, self.hidden_size)
    self.optim = None # init by calling self.init_optim()

  def init_optim(self):
    print('You should call this after loading model parameters')
    should_update = chain(self.encoder.parameters(), self.decoder.parameters())
    self.optim = optim.SGD(should_update, lr=0.01, momentum=0.9)

  def _inpt_for_encoder(self, nums):
    seq_len = -1 # Auto refer
    default_batch_size = 1 
    return self.embedding(t.LongTensor(nums)).reshape(seq_len, default_batch_size, self.input_size)


  # 每一步
  # dh, dc是loss的计算节点，联系着decoder + encoder
  # inpt不是计算节点，detached
  # correct_index: int， 理应输出的序号
  # for_select: (seq_size + 1, 1, hidden_size), detached
  def decode_and_train(self, dh, dc, inpt, correct_index, for_select):
    _,(next_dh, next_dc) = self.decoder(inpt, (dh,dc))
    # Calculate loss 
    # 先用一个大型linear层过一下, 然后直接跟for_select相乘即可
    # TODO: select的时候，用transformer的方案
    query = self.query_from_dh_layer(dh) # (1, 1, 9)
    # for_softmax = for_select dot query
    for_softmax = t.matmul(for_select.view(-1, self.hidden_size), query.view(self.hidden_size)).view(1, -1) # (1, seq_size+1)
    loss = self.CEL(for_softmax, t.LongTensor([correct_index])) # Have softmax
    # self.optim.zero_grad()
    # loss.backward()
    # self.optim.step()
    # print info
    print(f"Correct index: {correct_index}, Decoder output: {for_softmax.argmax()}")
    return next_dh, next_dc, loss

  def _h_c_or_file_symbols(self, index):
    return self.embedding(t.LongTensor([[index]])).detach()

  def _one_hot_labels_and_indexs(self, list_of_num):
    indexs = ordered_index(list_of_num, self.MAX_INT)
    indexs = list(map(lambda x: x + 1, indexs))
    indexs.append(0) # For EOF prepended
    one_hots = []
    for i in indexs:
      one_hot = t.zeros(len(indexs))
      one_hot[i] = 1
      one_hots.append(one_hot)
    return (one_hots, indexs)

  def train(self, list_of_num):
    # 转成inpts
    inpts = self._inpt_for_encoder(list_of_num.copy()).detach()
    # 喂进encoder(emb_of_EOF作为h0)，得到所有hidden_states as out & hn
    h0 = self._h_c_or_file_symbols(self.EOF)
    c0 = self._h_c_or_file_symbols(self.encoder_c0)
    out,(hn, _) = self.encoder(inpts, (h0, c0))
    # 将emb_of_EOF prepend到out，将out命名为for_select
    # 将for_select变成不需要grad。Encoder只通过hn来进行回溯
    for_select = t.cat((h0, out)).detach()

    # 通过经典排序，准备labels
    one_hot_labels, correct_indexs  = self._one_hot_labels_and_indexs(list_of_num.copy())

    next_dh = hn
    next_dc = self._h_c_or_file_symbols(self.decoder_c0)
    next_inpt = self._h_c_or_file_symbols(self.SOF)
    acc_loss = None
    print(f'Now, train for 0,{list_of_num}')
    for onehot, correct_index in zip(one_hot_labels, correct_indexs):
      # 将hn作为dh0，SOF作为dinpt0喂给decoder，得到dh1
      next_dh, next_dc, loss  = self.decode_and_train(next_dh, next_dc, next_inpt, correct_index, for_select)
      # Get next_inpt,
      next_inpt = for_select[correct_index].view(1, 1, self.hidden_size)
      # accumulate loss
      if acc_loss is None:
        acc_loss = loss
      else:
        acc_loss += loss
      print(f'now the loss is {acc_loss.item()}')
    # backward
    if self.optim is None:
      print('Trained failed! Because you have never init the optimizer!')
    else:
      self.optim.zero_grad()
      acc_loss.backward()
      self.optim.step()
    print(f'Trained {list_of_num}')

  def forward(list_of_num):
    pass


