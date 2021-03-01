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


