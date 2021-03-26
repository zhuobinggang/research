# BERT池化句子实验田
from exp4 import *
import bert_ja as B
import logging
import time
import utils as U
from importlib import reload

U.init_logger('exp5.log')

# start ======================= Loader Tested, No Touch =======================
class Loader():
  def __init__(self, ds, half, batch):
    self.half = ds.half = half
    self.ss_len = ds.ss_len = half * 2 + 1
    self.ds = self.dataset = ds
    self.batch = self.batch_size = batch
    self.start = self.start_point()

  def __iter__(self):
    return self

  def __len__(self):
    return self.end_point() - self.start_point() + 1

  def start_point(self):
    return 0

  def end_point(self):
    return len(self.ds.datas) - 1

  def get_data_by_index(self, idx):
    assert idx >= self.start_point()
    assert idx <= self.end_point()
    start = idx - self.half # 可能是负数
    ss, labels = self.ds[start] # 会自动切掉负数的部分
    correct_start = max(start, 0)
    pos = idx - correct_start
    return ss, labels, pos # 只需要中间的label

  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  # raise StopIteration()
  def __next__(self):
    start = self.start
    if start > self.end_point():
      self.start = self.start_point()
      raise StopIteration()
    else:
      results = []
      end = min(start + self.batch - 1, self.end_point())
      for i in range(start, end + 1):
        ss, label, pos = self.get_data_by_index(i)
        results.append((ss, label, pos))
      self.start = end + 1
      return results

  def shuffle(self):
    self.ds.shuffle()
# end ======================= Loader Tested, No Touch =======================

def handle_mass(mass):
  ss = []
  labels = []
  pos = []
  for s,l,p in mass:
    ss.append(s)
    labels.append(l[p])
    pos.append(p)
  return ss, labels, pos

def get_left_right_by_ss_pos(ss, pos):
  left = [ss[pos - 1], ss[pos]] if pos > 0 else [ss[pos]]
  right = [ss[pos], ss[pos + 1]] if pos < len(ss) - 1 else [ss[pos]]
  return left, right
 

class Model_Fuck(nn.Module):
  def __init__(self, weight_one = 1, hidden_size = 256, head = 8, dropout=0):
    super().__init__()
    self.max_memory_batch = 6
    self.hidden_size = hidden_size
    self.head = head
    self.dropout = dropout
    self.bert_size = 768
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, weight_one]))
    self.CEL = nn.CrossEntropyLoss()
    self.verbose = False
    self.init_bert()
    self.init_hook()
    self.optim = optim.AdamW(self.get_should_update(), self.learning_rate())
    print(f'Init AdamW with lr = {self.learning_rate()}')
    if GPU_OK:
      _ = self.cuda()

  def learning_rate(self):
    return 2e-5

  def init_hook(self):
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, 2)
    )

  def init_bert(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

  def set_verbose(self):
    self.verbose = not self.verbose

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {o.argmax(1).tolist()} Loss: {loss} ')

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters())

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    for ss, pos in zip(sss, poss):
      left, right = get_left_right_by_ss_pos(ss, pos)
      emb1 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb2 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      print(f'{emb1.shape[0]}, {emb2.shape[0]}')
      assert emb1.shape[0] == emb2.shape[0]
      mean = (emb1 + emb2) / 2 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    o = self.classifier(pooled_embs) # (batch, 2)
    loss = self.CEL(o, labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    for ss, pos in zip(sss, poss):
      left, right = get_left_right_by_ss_pos(ss, pos)
      emb1 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb2 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      assert emb1.shape[0] == emb2.shape[0]
      mean = (emb1 + emb2) / 2 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    o = self.classifier(pooled_embs) # (batch, 2)
    self.print_train_info(o, labels, -1)
    return o.argmax(1), labels

def init_G():
  G['ld'] = Loader(ds = data.train_dataset(ss_len = 3, max_ids = 64), half = 1, batch = 4)
  G['testld'] = Loader(ds = data.test_dataset(ss_len = 3, max_ids = 64), half = 1, batch = 4)
  G['devld'] = Loader(ds = data.dev_dataset(ss_len = 3, max_ids = 64), half = 1, batch = 4)

def get_datas(index, epoch, desc):
  return get_datas_org(index, epoch, G['m'], G['ld'], G['testld'], G['devld'], desc)

def train_simple(m, loader, epoch):
  logger = logging.debug
  loss_per_epoch = []
  loader.start = 0
  start = time.time()
  length = len(loader.ds.datas)
  for e in range(epoch):
    logger(f'start epoch{e}')
    total_loss = 0
    for mass in loader:
      logger(f'{loader.start}/{length}')
      total_loss += m.train(mass)
    avg_loss = total_loss / length
    loss_per_epoch.append(total_loss)
  end = time.time()
  logger(f'Trained! Epochs: {epoch}, Batch size: {loader.batch_size}, dataset length: {length}, Time cost: {end - start} seconds')
  return loss_per_epoch

def get_datas_org(index, epoch, m, ld, testld, devld,  desc='Nothing'):
  losses = train_simple(m, ld, epoch) # only one epoch for order matter model
  G[f'testdic_{index}'] = get_test_result_dic(m, testld)
  G[f'devdic_{index}'] = get_test_result_dic(m, devld)
  dic = {
    'testdic': G[f'testdic_{index}'],
    'devdic': G[f'devdic_{index}'],
  }
  R.request_my_logger(dic, desc)
  return losses

def get_test_result(m, loader):
  logger = logging.debug
  loader.start = 0
  start = time.time()
  length = len(loader.ds.datas)
  outputs = []
  targets = []
  for mass in loader:
    logger(f'TESTING: {loader.start}/{length}')
    out, labels = m.dry_run(mass)
    outputs += out.tolist()
    targets += labels.tolist()
  end = time.time()
  logger(f'TESTED! length={length} Time cost: {end - start} seconds')
  return outputs, targets

def get_test_result_dic(m, testld):
  testld.start = 0
  dic = {}
  if testld is None:
    dic['prec'] = -1
    dic['rec'] = -1
    dic['f1'] = -1
    dic['bacc'] = -1
  else: 
    outputs, targets = get_test_result(m, testld)
    prec, rec, f1, bacc = U.cal_prec_rec_f1_v2(outputs, targets)
    dic['prec'] = prec
    dic['rec'] = rec
    dic['f1'] = f1
    dic['bacc'] = bacc
  return dic

def run_test():
  init_G()
  G['ld'].ds.datas = G['ld'].ds.datas[:15]
  G['testld'].ds.datas = G['testld'].ds.datas[:15]
  G['devld'].ds.datas = G['devld'].ds.datas[:15]
  G['m'] = m = Model_Fuck()
  get_datas(0, 1, f'试着池化')
