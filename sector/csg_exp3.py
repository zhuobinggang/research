from csg_exp2 import * 
from transformers import BertModel, BertJapaneseTokenizer
import logging

class Dataset_Around_Split_Point(Dataset_Single_Sentence_True):
  def init_hook(self):
    print('DDDDDDDDDD')
    self.datas = []
    self.datas_processed = []
    self.datas_trimed = []
    self.tokens = 128
    self.max_size = 128
    self.init_toker()

  def process_datas(self, datas):
    self.datas = datas
    self.datas_processed = []
    length = len(datas)
    start = self.half_window_size - 1
    end = length - (self.half_window_size + 1)
    for i in range(start, end):
      l_and_r, label = self.__getitem_org__(i)
      self.datas_processed.append((l_and_r, label))

  def safe_get(self, datas, idx):
    return None if idx < 0 or idx > (len(datas) - 1) else datas[idx]

  def set_datas(self, datas):
    self.process_datas(datas)
    self.datas_trimed = []
    # Trim Redundant
    for i, (l_and_r, label) in enumerate(self.datas_processed):
      if label == 1:
        self.datas_trimed += [data for data in [self.safe_get(self.datas_processed, i-1), (l_and_r, label), self.safe_get(self.datas_processed, i+1)] if data is not None]

  def __len__(self):
    return len(self.datas_trimed)
    
  def __getitem__(self, idx):
    data = self.safe_get(self.datas_trimed, idx)
    if data is not None:
      (l,r), label = data
      l = self.joined(l) # string
      r = self.joined(r)
      l = self.token_encode(l)
      r = self.token_encode(r)
      return ([self.cls_id] + l + [self.sep_id] + r), label
    else:
      return None

class Train_DS_Around_Split_Point(Dataset_Around_Split_Point):
  def init_datas_hook(self):
    datas = data.read_trains()
    self.set_datas(datas)

def get_datas_trimed_redundant(test):
  # Get Datas for Single_Sentence_CSG
  ld = Loader_Single_Sentence_True(Train_DS_Around_Split_Point(2), 4)
  testld = Loader_Single_Sentence_True(Test_DS_Single_Sentence_True(2), 4)
  devld = Loader_Single_Sentence_True(Dev_DS_Single_Sentence_True(2), 4)
  if test:
    ld.dataset.datas = ld.dataset.datas[:30]
    testld.dataset.datas = testld.dataset.datas[:15]
    devld.dataset.datas = devld.dataset.datas[:15]
  mess = []
  for i in range(2 if test else 5):
    m = CSG.Model()
    loss = runner.train_simple(m, ld, 2)
    runner.logout_info(f'Trained Double_Sentence True model_{i}, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result(m, testld)
    devdic = runner.get_test_result(m, devld)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['datas_trimed_redundant_mess'] = mess

# ========================

class BERT_SEGBOT(CSG.Model):
  def init_hook(self):
    self.flatten_then_sigmoid = nn.Sequential(
      nn.Linear(self.hidden_size * 2, 1),
      nn.Sigmoid(),
    )
    self.bi_gru_batch_first = nn.GRU(self.bert_size, self.hidden_size, batch_first=True, bidirectional=True)
    self.EOF = t.randn(1, self.bert_size)
    if GPU_OK:
      self.EOF = self.EOF.cuda()
    self.CEL = nn.CrossEntropyLoss()

  def get_should_update(self):
    return chain(self.bert.parameters(), self.flatten_then_sigmoid.parameters(), self.bi_gru_batch_first.parameters())

  # ss: (sentence_size, 768)
  def integrate_sentences_info(self, ss):
    ss = t.cat((ss, self.EOF)) # (sentence_size + 1, 768)
    ss = ss.view(1, ss.shape[0], ss.shape[1]) # (1, sentence_size + 1, 768)
    out, _ = self.bi_gru_batch_first(ss) # (1, sentence_size + 1, 2 * hidden_size)
    out = out.view(out.shape[1], out.shape[2])
    return out

  # embs: (sentence_size + 1, 2 * hidden_size)
  def flatten_then_softmax(self, embs):
    return self.sigmoid(self.flatten_layer(embs)) # (sentence_size + 1, 1)

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {o.argmax().item()} Loss: {loss} ')

  # inpts: token_ids, attend_marks
  # token_ids: (sentence_size, max_id_len)
  # labels: (sentence_size + 1)
  def train(self, inpts, labels):
    token_ids, attend_marks = inpts # token_ids = attend_marks: (sentence_size, max_id_len)
    labels = t.LongTensor([labels.tolist().index(1)]) # (1)
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      labels = labels.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (sentence_size, 768)
    embs = self.processed_embs(embs) # (sentence_size, 768)

    o = self.integrate_sentences_info(embs) # (sentence_size + 1, 2 * hidden_size)
    o = self.flatten_then_sigmoid(o) # (sentence_size + 1, 1)
    loss = self.CEL(o.view(1, -1), labels)

    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())

    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    token_ids, attend_marks = inpts # token_ids = attend_marks: (sentence_size, max_id_len)
    org_labels = labels
    labels = t.LongTensor([labels.tolist().index(1) if labels is not None else -1])  # (1)
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      labels = labels.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (sentence_size, 768)
    embs = self.processed_embs(embs) # (sentence_size, 768)

    o = self.integrate_sentences_info(embs) # (sentence_size + 1, 2 * hidden_size)
    o = self.flatten_then_sigmoid(o) # (sentence_size + 1, 1)
    self.print_train_info(o, org_labels, -1)
    return o.argmax().item()


class Dataset_Segbot(t.utils.data.dataset.Dataset):
  def __init__(self, ss_len = 8, max_ids = 64):
    super().__init__()
    self.ss_len = ss_len
    self.max_ids = max_ids
    self.init_datas_hook()
    self.init_toker()
    self.init_hook()
    self.start = 0

  def init_hook(self):
    pass

  def count_exceed_length_ground_truth(self):
    counter = 0
    for _, labels in self.ground_truth_datas:
      if labels.index(1) >= self.ss_len:
        counter += 1
    return counter

  def init_datas_hook(self):
    self.datas = []
    self.ground_truth_datas = []

  def set_datas(self, datas):
    self.datas = datas

  def is_begining(self, s):
    return s.startswith('\u3000')
  
  def next_start(self, start):
    start = start + 1
    end = len(self.datas)
    next_start = end
    for i in range(start, end):
      s = self.datas[i]
      if self.is_begining(s):
        next_start = i
        break
    return next_start
        
  def no_indicator(self, ss):
    return [s.replace('\u3000', '') for s in ss]

  def get_ss_and_labels(self, start):
    end = min(start + self.ss_len, len(self.datas)) 
    ss = []
    cut_point = -1
    for i in range(start, end):
      s = self.datas[i]
      ss.append(s)
      if i != start and cut_point == -1  and self.is_begining(s):
        cut_point = len(ss) - 1
    ss = self.no_indicator(ss)
    labels = np.zeros(len(ss) + 1, np.int8).tolist()
    if cut_point != -1:
      labels[cut_point] = 1
    else: 
      labels[-1] = 1
    return ss, labels

  def __getitem__(self, start):
    ss, labels = self.get_ss_and_labels(start)
    ids_and_masks = [self.token_encode_with_masks(s) for s in ss]
    return ([ids for ids, masks in ids_and_masks], [masks for ids, masks in ids_and_masks]), labels

  def __len__(self):
    return len(self.datas)

  def shuffle(self):
    random.shuffle(self.datas)

  def token_encode_with_masks(self, text):
    ids = self.toker.encode(text, add_special_tokens = False)
    masks = None
    if self.max_ids < len(ids):
      # logging.warning(f'Length {len(ids)} exceed!: {text[:30]}...')
      ids = [self.cls_id] + ids[0: self.max_ids + 1]
      masks = np.ones(len(ids), dtype=np.int8).tolist()
    else:
      zeros = np.zeros(self.max_ids - len(ids), np.int8).tolist()
      ids = [self.cls_id] + ids + [self.sep_id] + zeros
      masks = np.ones(len(ids) - len(zeros), dtype=np.int8).tolist() + zeros
    return ids, masks

  def init_toker(self):
    toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.toker = toker
    self.pad_id = toker.pad_token_id
    self.cls_id = toker.cls_token_id
    self.sep_id = toker.sep_token_id
    self.sentence_period = '。'

class Test_DS_Segbot(Dataset_Segbot):
  def init_hook(self):
    datas = data.read_tests()
    self.set_datas(datas)
  def __getitem__(self, idx):
    (ids, ms), labels = super().__getitem__(idx)
    return (t.LongTensor(ids), t.LongTensor(ms)), t.LongTensor(labels)
    

class Dev_DS_Segbot(Dataset_Segbot):
  def init_hook(self):
    datas = data.read_devs()
    self.set_datas(datas)
  def __getitem__(self, idx):
    (ids, ms), labels = super().__getitem__(idx)
    return (t.LongTensor(ids), t.LongTensor(ms)), t.LongTensor(labels)

class Train_DS_Segbot(Dataset_Segbot):
  def init_hook(self):
    datas = data.read_trains()
    self.set_datas(datas)

  def set_datas(self, datas):
    self.datas = datas
    self.ground_truth_datas = []
    start = 0
    stop = False
    while start < len(self.datas):
      inpts, labels = self[start]
      self.ground_truth_datas.append((inpts, labels))
      start = self.next_start(start)

class Loader_Segbot_GroundTrue():
  def __init__(self, ds):
    self.ds = ds
    self.dataset = ds
    self.start = 0
    self.batch_size = self.ds.ss_len

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.ds.ground_truth_datas)

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start >= len(self):
      self.start = 0
      raise StopIteration()
    else:
      (ids, masks), labels = self.ds.ground_truth_datas[self.start]
      self.start += 1
      return (t.LongTensor(ids), t.LongTensor(masks)), t.LongTensor(labels)

class Loader_Segbot_Fool():
  def __init__(self, ds):
    self.ds = ds
    self.dataset = ds
    self.start = 0
    self.batch_size = self.ds.ss_len

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.ds.ground_truth_datas)

  def __getitem__(self, idx):
    (ids, ms), labels = self.ds[idx]
    return (t.LongTensor(ids), t.LongTensor(ms)), t.LongTensor(labels)


class Loader_Segbot_Normal():
  def __init__(self, ds):
    self.ds = ds
    self.dataset = ds
    self.start = 0
    self.batch_size = self.ds.ss_len

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.ds.ground_truth_datas)

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start >= len(self):
      self.start = 0
      raise StopIteration()
    else:
      (ids, masks), labels = self.ds.ground_truth_datas[self.start]
      self.start += 1
      return (t.LongTensor(ids), t.LongTensor(masks)), t.LongTensor(labels)

 
# =======================

def get_datas_segbot(test):
  ld = Loader_Segbot_GroundTrue(Train_DS_Segbot(ss_len = 16))
  testds = Test_DS_Segbot(ss_len = 16)
  devds = Dev_DS_Segbot(ss_len = 16)
  if test:
    ld.dataset.ground_truth_datas = ld.dataset.ground_truth_datas[:3]
    testds.datas = testds.datas[:30]
    devds.datas = devds.datas[:30]
  mess = []
  for i in range(2 if test else 5):
    m = BERT_SEGBOT(hidden_size = 256)
    m.verbose = True
    loss = runner.train_simple(m, ld, 2) # only one epoch for order matter model
    runner.logout_info(f'Trained order matter model_{i} only one epoch, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result_segbot(m, testds)
    devdic = runner.get_test_result_segbot(m, devds)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['segbot_mess'] = mess


def run(test = False):
  # get_datas_trimed_redundant(test)
  get_datas_segbot(test)
