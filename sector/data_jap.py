import random
from sqlitedict import SqliteDict
from sentence_transformers import SentenceTransformer
from sentence_transformers import models
import numpy as np

db = None
model = None

def get_sbert():
  global model
  if model is None:
    transformer = models.BERT('cl-tohoku/bert-base-japanese-whole-word-masking')
    pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[transformer, pooling])
  return model
    


def read_docs(data_id = 1):
  if data_id == 1:
    filenames = ['sansirou', 'sorekara', 'mon', 'higan', 'gyoujin'] 
  elif data_id == 2:
    filenames = ['kokoro'] 
  elif data_id == 3:
    filenames = ['meian'] 
  paths = [f'datasets/{name}_new.txt' for name in filenames]
  docs = []
  for path in paths:
    with open(path) as f:
      lines = f.readlines()
      docs.append(lines)
  return docs

def read_lines(data_id = 1):
  docs = read_docs(data_id)
  lines = []
  for doc in docs:
    lines += doc
  return [line.replace(' ', '').replace('\n', '') for line in lines]

def read_sentences(data_id = 1):
  lines = read_lines(data_id)
  sentences = []
  for line in lines:
    ss = line.split('。')
    for s in ss:
      if len(s) > 1:
        sentences.append(s)
  return sentences

def read_trains():
  return read_sentences(1)

def read_tests():
  return read_sentences(3)

def read_devs():
  return read_sentences(2)

def no_indicator(ss):
  return [s.replace('\u3000', '') for s in ss]

class Dataset():
  def __init__(self, half_window_size = 1):
    self.init_datas_hook()
    self.half_window_size = half_window_size

  def init_datas_hook(self):
    self.datas = []

  def __len__(self):
    return len(self.datas) - 1

  def __getitem__(self, idx):
    if idx >= len(self.datas) - 1:
      print(f'Warning: Should not get idx={idx}')
      return None
    left = []
    start = max(0, idx + 1 - self.half_window_size)
    end = min(idx + 1, len(self.datas))
    for i in range(start, end):
      left.append(self.datas[i])
    right = []
    start = max(0, idx + 1)
    end = min(idx + 1 + self.half_window_size, len(self.datas))
    for i in range(start, end):
      right.append(self.datas[i])
    label = 1 if right[0].startswith('\u3000') else 0
    left = no_indicator(left)
    right = no_indicator(right)
    return (left,right), label

  def shuffle(self):
    random.shuffle(self.datas)

class Train_DS(Dataset):
  def init_datas_hook(self):
    self.datas = read_trains()

class Test_DS(Dataset):
  def init_datas_hook(self):
    self.datas = read_tests()

class Dev_DS(Dataset):
  def init_datas_hook(self):
    self.datas = read_devs()

class Train_DS_Mini(Dataset):
  def init_datas_hook(self):
    self.datas = read_trains()[:100]

class Test_DS_Mini(Dataset):
  def init_datas_hook(self):
    self.datas = read_tests()[:50]


class Loader():
  def __init__(self, ds, batch_size = 4):
    self.start = 0
    self.ds = ds
    self.batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self):
    if self.start == len(self.ds):
      self.start = 0
      raise StopIteration()
    results = []
    end = min(self.start + self.batch_size, len(self.ds))
    for i in range(self.start, end):
      results.append(self.ds[i])
    self.start = end
    return [d[0] for d in results], [d[1] for d in results]

  def shuffle(self):
    self.ds.shuffle()


# return: (768)
def get_emb(s):
  sbert = get_sbert()
  return sbert.encode(s)

def get_db():
  global db
  if db is None:
    default_sbert_db = './datasets/natume_soseki.sqlite'
    db = SqliteDict(default_sbert_db, autocommit=True)
    print(f'Inited db using {default_sbert_db}')
  return db

# return: (768); numpy array
def get_cached_emb(s):
  db = get_db()
  array_str = db.get(s)
  if array_str is not None:
    # print(f'Cached: {s}')
    return __string_to_numpy(array_str)
  else:
    print(f'Not cached: {s}')
    nparray = get_emb(s)
    db[s] = __numpy_to_string(nparray)
    return nparray

# return (?, 768)
def cached_ss2embs(ss):
  return [get_cached_emb(s) for s in ss]


def precache():
  ss_trian = read_trains()
  ss_test = read_tests()
  ss = ss_trian + ss_test
  for i, s in enumerate(ss):
    _ = get_cached_emb(s)
    print(f'{i}/{len(ss)}')


def __numpy_to_string(A):
  return A.tobytes().hex()

def __string_to_numpy(S):
  return np.frombuffer(bytes.fromhex(S), dtype=np.float32)


