import torch as t

input_range = [0,2020] # min, max
embedding_dim = 9 # Number embedding dimension
ember = None # Torch embedding layer

def neural_sort(nums):
  print(sorted(nums))

# input: [5,3,2,6] //number(int) sequence
def embeddings_from_ints(nums):
  global ember, input_range, embedding_dim
  # Check for ember
  if ember is None:
    ember = t.nn.Embedding(input_range[1], embedding_dim)
  else:
    pass
  # turn nums to embeddings
  ts = t.LongTensor(nums)
  return ember(ts)


def encode(embs):
  # LSTM embs and get output
  pass



