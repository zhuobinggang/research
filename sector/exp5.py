# BERT池化句子实验田
from exp4 import *
import bert_ja as B
import logging
import time
import utils as U
from importlib import reload

U.init_logger('exp5.log')
GPU_OK = t.cuda.is_available()

Loader = data.Loader

def handle_mass(mass):
  ss = []
  labels = []
  pos = []
  for s,l,p in mass:
    ss.append(s)
    labels.append(l[p])
    pos.append(p)
  return ss, labels, pos

def get_left_right_by_ss_pos(ss, pos, distant = 1):
  # left = [ss[pos - 1], ss[pos]] if pos > 0 else [ss[pos]]
  left_index = pos - distant
  if left_index >= 0:
    left = [ss[left_index], ss[pos]]
  else:
    left = [ss[pos]]
  right_index = pos + distant
  if right_index <= (len(ss) - 1):
    right = [ss[pos], ss[right_index]]
  else:
    right = [ss[pos]]
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
      # print(f'{emb1.shape[0]}, {emb2.shape[0]}')
      assert emb1.shape[0] == emb2.shape[0]
      mean = (emb1 + emb2) / 2 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
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
    if GPU_OK:
      labels = labels.cuda()
    o = self.classifier(pooled_embs) # (batch, 2)
    self.print_train_info(o, labels, -1)
    return o.argmax(1), labels


class Model_Fuck_2vs2(Model_Fuck):
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    for ss, pos in zip(sss, poss):
      left, right = get_left_right_by_ss_pos(ss, pos)
      emb1 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb2 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      left, right = get_left_right_by_ss_pos(ss, pos, distant = 2)
      emb3 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb4 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      # print(f'{emb1.shape[0]}, {emb2.shape[0]}')
      assert emb1.shape[0] == emb2.shape[0] == emb3.shape[0] == emb4.shape[0]
      mean = (emb1 + emb2 + emb3 + emb4) / 4 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
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
      left, right = get_left_right_by_ss_pos(ss, pos, distant = 2)
      emb3 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb4 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      # print(f'{emb1.shape[0]}, {emb2.shape[0]}')
      assert emb1.shape[0] == emb2.shape[0] == emb3.shape[0] == emb4.shape[0]
      mean = (emb1 + emb2 + emb3 + emb4) / 4 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
    o = self.classifier(pooled_embs) # (batch, 2)
    self.print_train_info(o, labels, -1)
    return o.argmax(1), labels


class Model_Baseline(Model_Fuck):
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    clss = []
    for ss, pos in zip(sss, poss): 
      cls = B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, pos) # (784)
      clss.append(cls)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
    clss = t.stack(clss) # (batch, 784)
    o = self.classifier(clss) # (batch, 2)
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
    clss = []
    for ss, pos in zip(sss, poss): 
      cls = B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, pos) # (784)
      clss.append(cls)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
    clss = t.stack(clss) # (batch, 784)
    o = self.classifier(clss) # (batch, 2)
    self.print_train_info(o, labels, -1)
    return o.argmax(1), labels 

def init_G(half = 1):
  G['ld'] = Loader(ds = data.train_dataset(ss_len = half * 2 + 1, max_ids = 64), half = half, batch = 4)
  G['testld'] = Loader(ds = data.test_dataset(ss_len = half * 2 + 1, max_ids = 64), half = half, batch = 4)
  G['devld'] = Loader(ds = data.dev_dataset(ss_len = half * 2 + 1, max_ids = 64), half = half, batch = 4)

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
  init_G(2)
  G['ld'].ds.datas = G['ld'].ds.datas[:15]
  G['testld'].ds.datas = G['testld'].ds.datas[:15]
  G['devld'].ds.datas = G['devld'].ds.datas[:15]
  G['m'] = m = Model_Fuck_2vs2()
  get_datas(0, 1, f'池化test 2:2')

def run():
  init_G(2)
  G['m'] = m = Model_Fuck_2vs2()
  get_datas(0, 1, f'池化压测 2: 2: epoch0')
  get_datas(1, 1, f'池化压测 2: 2: epoch1')
  get_datas(2, 1, f'池化压测 2: 2: epoch2')
  for i in range(5):
    base = 10
    G['m'] = m = Model_Fuck_2vs2()
    get_datas(base + i, 2, f'池化2:2, 跑5次，每次2epoch')


def run_len1():
  init_G()
  G['m'] = m = Model_Fuck()
  get_datas(0, 1, f'池化epoch压测: epoch1')
  get_datas(1, 1, f'池化epoch压测: epoch2')
  get_datas(2, 1, f'池化epoch压测: epoch3')
  for i in range(4):
    base = 10
    G['m'] = m = Model_Fuck()
    get_datas(base + i, 2, f'试着池化, 跑四次，每次2epoch')


def run_baseline_1vs2():
  init_G(1)
  G['m'] = m = Model_Baseline()
  get_datas(0, 1, f'Baseline 1:2')
  get_datas(1, 1, f'Baseline 1:2')
