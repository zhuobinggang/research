import torch as t

input_range = [0,2000] # min, max
num_embeddings = 2020
# embedding_dim = 9 # Number embedding dimension
input_size = 9
hidden_size = input_size
ember = None # Torch embedding layer
encoder = None
decoder = None
# 2000以上的数字用于特殊用途
SOF = 2001
EOF = 2002

def neural_sort(nums):
  print(sorted(nums))

# input: [5,3,2,6] //number(int) sequence
# output: emb([0,5,3,2,6])
def embeddings_from_ints(nums):
  global ember, input_range, input_size
  # Check for ember
  if ember is None:
    ember = t.nn.Embedding(num_embeddings, input_size)
  else:
    pass
  # turn nums to embeddings
  ts = t.LongTensor(nums)
  return ember(ts)

# input: (seq_len, input_size)
def add_fake_batch(embs_of_nums):
  seq_len, inpt_size = embs_of_nums.shape
  return embs_of_nums.reshape(seq_len, 1, inpt_size)

def inpt_for_lstm(nums):
  return add_fake_batch(embeddings_from_ints(nums))

# output: [[input_size]]
def SOF_embedded():
  global ember
  if ember is None:
    print('You should prepare an ember before using this function')
  return ember(t.LongTensor([SOF]))

def EOF_embedded():
  global ember
  if ember is None:
    print('You should prepare an ember before using this function')
  return ember(t.LongTensor([EOF]))

# output: (out, (hn, cn))
def encode(inpt):
  global encoder
  # create encoder if not ready
  if encoder is None:
    encoder = t.nn.LSTM(input_size, hidden_size)
  else:
    pass
  # EOF as h0
  h0 = add_fake_batch(EOF_embedded())
  c0 = t.randn(1,1, hidden_size)
  return encoder(inpt, (h0, c0))


def first_step_decode(hn):
  h0 = hn
  c0 = t.randn(1,1, hidden_size)
  return step_decode(add_fake_batch(SOF_embedded()), h0, c0)

def after_first_step(hn):
  _,(h1,_) = first_step_decode(hn)
  
def step_decode(inpt, h0, c0):
  global decoder 
  if decoder is None:
    decoder = t.nn.LSTM(input_size, hidden_size)
  else:
    pass
  return decoder(inpt, (h0, c0))

# TODO: 
def index_from_hidden_state_by_argmax(hidden_state, inpt_embs, SOF_emb):
  pass
  
def decode():
  pass

# inpt: (batch, input_size)
def wrap_fake_seq(inpt):
  batch, input_size = inpt.shape
  return inpt.reshape(1, batch, input_size)

def test():
  inpt = inpt_for_lstm([5,3,2])
  _, (hn, _) = encode(inpt)
  _, (dh1, _) = first_step_decode(hn)
  layer_dh_to_query = t.nn.Linear(hidden_size, input_size)
  query = layer_dh_to_query(dh1)
  # concat input with EOF
  eof = add_fake_batch(EOF_embedded())
  inpts_for_attend = t.cat((eof, inpt))
  # all inputs dot query then softmax
  seq_vs_similarity = t.mm(inpts_for_attend.view(-1, input_size), query.view(input_size, 1))
  result_after_softmax = t.softmax(seq_vs_similarity.view(-1),0)
  correct_result = t.FloatTensor([0,0,0,1])
  temp = t.sum(t.abs(correct_result - result_after_softmax))
  loss = 0.5 * temp * temp
  # TODO: backward
  # 假设只要调用loss的backward就能正确backward，接下来是判断有没有下一步
  # 首先判断labels有没有到头
  if have_no_next_label():
    # 什么也不干
    pass
  else:
    # 用正确的输出作为输入
    next_step(correct_output)

def ordered_index(list_of_num):
  MAX_INT = 99999
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


def one_hot_labels(list_of_num):
  indexs = ordered_index(list_of_num)
  indexs = list(map(lambda x: x + 1, indexs))
  indexs.append(0) # For EOF prepended
  one_hots = []
  for i in indexs:
    one_hot = t.zeros(len(indexs))
    one_hot[i] = 1
    one_hots.append(one_hot)
  return one_hots

