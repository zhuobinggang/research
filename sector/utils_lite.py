import numpy as np
import torch
import requests
t = torch
GPU_OK = t.cuda.is_available()

def position_encoding_ddd(t, i, d):
  k = int(i/2)
  omiga = 1 / np.power(10000, 2 * k / d)
  even = (i / 2).is_integer()
  return np.sin(omiga * t) if even else np.cos(omiga * t)

# seq: (seq_len, feature)
# return: (seq_len, feature)
def position_encoding(seq):
  embs = []
  for t, data in enumerate(seq):
    d = data.shape[0]
    pos_emb = [position_encoding_ddd(t, i, d) for i in range(0, d)]
    pos_emb = torch.tensor(pos_emb)
    embs.append(pos_emb)
  embs = torch.stack(embs)
  return embs.cuda() if GPU_OK else embs

def position_matrix(seq_len, feature):
  d = feature
  embs = []
  for j in range(seq_len):
    pos_emb = [position_encoding_ddd(j, i, d) for i in range(d)]
    pos_emb = torch.tensor(pos_emb)
    embs.append(pos_emb)
  embs = torch.stack(embs)
  return embs.cuda() if GPU_OK else embs

def request_my_logger(dic, desc = 'No describe'):
  try:
    url = "https://hookb.in/b9xlr2GnnjC3DDogQ0jY"
    dic['desc'] = desc
    requests.post(url, json=dic)
  except:
    print('Something went wrong in request_my_logger()')
