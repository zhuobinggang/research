import torch as t

input_range = [0,2020] # min, max
# embedding_dim = 9 # Number embedding dimension
input_size = 9
hidden_size = 12
ember = None # Torch embedding layer
lstm = None # Encoder

def neural_sort(nums):
  print(sorted(nums))

# input: [5,3,2,6] //number(int) sequence
def embeddings_from_ints(nums):
  global ember, input_range, input_size
  # Check for ember
  if ember is None:
    ember = t.nn.Embedding(input_range[1], input_size)
  else:
    pass
  # turn nums to embeddings
  ts = t.LongTensor(nums)
  return ember(ts)

# input: (seq_len, inpt_size)
def add_fake_batch(embs_of_nums):
  seq_len, inpt_size = embs_of_nums.shape
  return embs_of_nums.reshape(seq_len, 1, inpt_size)

# TODO: Q: Should I output all [h] for attention?
def encode(inpt):
  if lstm is None:
    lstm = t.nn.LSTM(input_size, hidden_size)
  else:
    pass
  # Make h0,c0, and feed in LSTM
  h0 = t.randn(1,1, hidden_size)
  c0 = t.randn(1,1, hidden_size)
  return lstm(inpt, (h0, c0))
  
def decode():
  pass 


# inpt: (batch, input_size)
def wrap_fake_seq(inpt):
  batch, input_size = inpt.shape
  return inpt.reshape(1, batch, input_size)

def test_lstm_every_step():
  inpt = add_fake_batch(embeddings_from_ints([5, 3, 2]))
  lstm = t.nn.LSTM(input_size, hidden_size)
  h0 = t.randn(1,1, hidden_size)
  c0 = t.randn(1,1, hidden_size)
  out,(hn, cn) = lstm(inpt, (h0, c0))
  # By step
  inpt0 = wrap_fake_seq(inpt[0])
  inpt1 = wrap_fake_seq(inpt[1])
  inpt2 = wrap_fake_seq(inpt[2])
  out1, (h1, c1) = lstm(inpt0, (h0, c0))
  out2, (h2, c2) = lstm(inpt1, (h1, c1))
  out3, (h3, c3) = lstm(inpt2, (h2, c2))
  print('Please check that they are the same')
  print(out3)
  print(out[2])
  print(h3)
  


