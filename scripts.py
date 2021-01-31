# === db cache sentence embedding ===
import torch as t
from sqlitedict import SqliteDict
default_sbert_db = './datasets/sbert_stsb_distilbert_base.sqlite'
db = None

def get_cached_embedding(s):
  global db
  if db is None:
    db = SqliteDict(default_sbert_db, autocommit=True)
    print(f'Inited db using {default_sbert_db}')
  array_str = db.get(s)
  if array_str is not None:
    return t.from_numpy(__string_to_numpy(array_str))
  else:
    print(f'Not cached: {s}')
    tensor = get_embedding(s)
    db[s] = __numpy_to_string(tensor.numpy())
    return tensor
    
def __numpy_to_string(A):
  return A.tobytes().hex()

def __string_to_numpy(S):
  return np.frombuffer(bytes.fromhex(S), dtype=np.float32)

# === Dataset & Loader ===

class MyDataset():
  def __init__(self):
    self.init_datas_hook()

  def init_datas_hook(self):
    self.datas = []

  def __len__(self):
    return len(self.datas)

  def __getitem__(self, idx):
    return self.datas[idx][0], self.datas[idx][1]

  def shuffle(self):
    random.shuffle(self.datas)
    
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
