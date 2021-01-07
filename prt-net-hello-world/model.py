import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import random
import time

def print_table(results, step):
  counter = 0
  for accuracy, ex_rate, short_rate, repeat_rate in results:
    counter += step
    print(f'|{counter}|{round(accuracy, 2)}|{round(ex_rate, 2)}|{round(short_rate, 2)}|{round(repeat_rate, 2)}|')


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

  def _define_variables(self, hidden_state_size):
    self.input_range = [0,100] # min, max
    self.num_embeddings = 120
    # embedding_dim = 9 # Number embedding dimension
    self.input_size = hidden_state_size
    self.hidden_size = hidden_state_size
    # 2000以上的数字用于特殊用途
    self.SOF = 101
    self.EOF = 102
    self.encoder_c0 = 103
    self.decoder_c0 = 104
    # others
    self.MAX_INT = 999999
    self.max_decode_length = 10 # 以防超量输出

  def __init__(self, hidden_state_size = 50):
    super().__init__()
    self._define_variables(hidden_state_size)
    self.embedding = nn.Embedding(self.num_embeddings, self.input_size)
    self.embedding.requires_grad_(False)#TODO: Do not add to optim
    self.encoder = t.nn.LSTM(self.input_size, self.hidden_size)
    self.decoder = t.nn.LSTM(self.input_size, self.hidden_size)
    self.CEL = nn.CrossEntropyLoss()
    self.query_from_dh_layer = nn.Linear(self.hidden_size, self.hidden_size)
    self.optim = None # init by calling self.init_optim()

  def init_optim(self):
    print('You should call this after loading model parameters')
    should_update = chain(self.encoder.parameters(), self.decoder.parameters(), self.query_from_dh_layer.parameters())
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
    query = self.query_from_dh_layer(dh) # (1, 1, hidden_size)
    # for_softmax = for_select dot query
    for_softmax = t.matmul(for_select.view(-1, self.hidden_size), query.view(self.hidden_size)).view(1, -1) # (1, seq_size+1)
    loss = self.CEL(for_softmax, t.LongTensor([correct_index])) # Have softmax
    # print info
    result_index = for_softmax.argmax()
    # print(f"Correct index: {correct_index}, Decoder output: {result_index}")
    return next_dh, next_dc, loss, (correct_index, result_index)

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

  def SGD_train(self, list_of_list_of_num, epoch = 5):
    if self.optim is None:
      print('Failed! You should init optim at first! Just call init_optim()!')
    else: 
      # print(f'Start train, epoch = {epoch}')
      list_of_list_of_num = list_of_list_of_num.copy()
      for i in range(epoch):
        # print(f'Start epoch{i}')
        random.shuffle(list_of_list_of_num)
        for list_of_num in list_of_list_of_num:
          self.train(list_of_num)
      print('Trained!')

  def SGD_train_output_table(self, train_datas, test_datas, epoch = 5, step = 5):
    counter = 0
    results = []
    start = time.time()
    print(f'Start train, epoch = {epoch}')
    for i in range(epoch):
      print(f'Start epoch{i}')
      counter += 1
      if counter % step  == 0:
        self.SGD_train(train_datas, step)
        results.append(self.test(test_datas))
    print_table(results, step)
    end = time.time()
    print(f'Epoch count: {epoch}, Train time: {end - start} seconds')
    return results

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
    # print(f'Now, train for 0,{list_of_num}')
    for onehot, correct_index in zip(one_hot_labels, correct_indexs):
      # 将hn作为dh0，SOF作为dinpt0喂给decoder，得到dh1
      next_dh, next_dc, loss, (_, ouput_index)  = self.decode_and_train(next_dh, next_dc, next_inpt, correct_index, for_select)
      # Get next_inpt,
      # next_inpt = for_select[correct_index].view(1, 1, self.hidden_size)
      next_inpt = for_select[ouput_index].view(1, 1, self.hidden_size)
      # accumulate loss
      if acc_loss is None:
        acc_loss = loss
      else:
        acc_loss += loss
      # print(f'now the loss is {acc_loss.item()}')
    # backward
    if self.optim is None:
      print('Trained failed! Because you have never init the optimizer!')
    else:
      self.optim.zero_grad()
      acc_loss.backward()
      self.optim.step()
    # print(f'Trained {list_of_num}')

  def forward(self, list_of_num):
    pass

  # next_inpt, next_dh, next_dc, result_index = self.decode(next_dh, next_dc, next_inpt, for_select)
  def decode(self, dh, dc, inpt, for_select):
    _,(next_dh, next_dc) = self.decoder(inpt, (dh,dc))
    # TODO: select的时候，用transformer的方案
    query = self.query_from_dh_layer(dh) # (1, 1, 9)
    # for_softmax = for_select dot query
    for_softmax = t.matmul(for_select.view(-1, self.hidden_size), query.view(self.hidden_size)).view(1, -1) # (1, seq_size+1)
    # print info
    result_index = for_softmax.argmax().item()
    next_inpt = for_select[result_index].view(1, 1, self.hidden_size)
    return next_inpt, next_dh, next_dc, result_index


  @t.no_grad()
  def dry_run(self, list_of_num):
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

    result_indexs = []
    # Decoder dry run
    for current_step_num in range(self.max_decode_length):
      next_inpt, next_dh, next_dc, result_index = self.decode(next_dh, next_dc, next_inpt, for_select)
      result_indexs.append(result_index)
      if result_index == 0:
        break
      else:
        pass
    
    return correct_indexs, result_indexs

  def test(self, list_of_list_of_num):
    total_num = len(list_of_list_of_num)
    # correct rate = correct_times / try_times
    correct_times = 0
    try_times = 0
    # length exceed rate = length_exceed_num / total_num
    length_exceed_num = 0
    # length shorted rate = length_shorted_num / total_num
    length_shorted_num = 0
    # repeat rate = repeat_num / total_num
    repeat_num = 0

    for list_of_num in list_of_list_of_num:
      (correct_indexs, result_indexs) = self.dry_run(list_of_num)
      result_nums = list(map(lambda i: list_of_num[i-1], filter(lambda x: x>0, result_indexs)))
      print(f'origin nums: {list_of_num}, result nums: {result_nums}')
      # print(f'correct indexs: {correct_indexs}, result indexs: {result_indexs}')
      length_exceed_num += 1 if len(result_indexs) > len(correct_indexs) else 0
      length_shorted_num += 1 if len(result_indexs) < len(correct_indexs) else 0
      try_times += len(result_indexs)
      for correct_index, result_index in zip(correct_indexs, result_indexs):
        correct_times += 1 if correct_index == result_index else 0
      repeat_num += 1 if len(result_indexs) != len(set(result_indexs)) else 0
    
    correct_rate = correct_times / try_times
    length_exceed_rate = length_exceed_num / total_num
    length_shorted_rate = length_shorted_num / total_num
    repeat_rate = repeat_num / total_num
    return correct_rate, length_exceed_rate, length_shorted_rate, repeat_rate

      
def run_example():
  m = Model()
  m.init_optim()
  train_datas = [] # Using data_generator.read_data() to read train.txt 
  test_datas = [] # Using data_generator.read_data() to read test.txt 
  m.SGD_train(train_datas, 10)
  m.test(test_datas)

def save(m):
  path = f'save/model_{m.hidden_size}.tch' 
  t.save(m, path)

def load(hidden_size = 50):
  path = f'save/model_{hidden_size}.tch' 
  return t.load(path)
