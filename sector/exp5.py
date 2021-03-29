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
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.fl_rate = 0

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

  def cal_loss(self, out, labels):
    assert len(labels.shape) == 1
    assert len(out.shape) == 2
    assert labels.shape[0] == out.shape[0]
    total = []
    for o, l in zip(out, labels):
      pt = o if (l == 1) else (1 - o)
      loss = (-1) * t.log(pt) * t.pow((1 - pt), self.fl_rate)
      total.append(loss)
    total = t.stack(total)
    return total.sum()

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
    o = self.classifier(pooled_embs) # (batch, 1)
    loss = self.cal_loss(o, labels)
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
    o = self.classifier(pooled_embs) # (batch, 1)
    self.print_train_info(o, labels, -1)
    return fit_sigmoided_to_label(o), labels


# 基于2:1池化使用FL
class Model_Fuck_FL(Model_Fuck):
  def init_hook(self):
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.fl_rate = 0

  def cal_loss(self, out, labels):
    assert len(labels.shape) == 1
    assert len(out.shape) == 2
    assert labels.shape[0] == out.shape[0]
    total = []
    for o, l in zip(out, labels):
      pt = o if (l == 1) else (1 - o)
      loss = (-1) * t.log(pt) * t.pow((1 - pt), self.fl_rate)
      total.append(loss)
    total = t.stack(total)
    return total.sum()

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
    o = self.classifier(pooled_embs) # (batch, 1)
    loss = self.cal_loss(o, labels)
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
    o = self.classifier(pooled_embs) # (batch, 1), sigmoided
    self.print_train_info(o, labels, -1)
    return fit_sigmoided_to_label(o), labels


# 不用什么left right了，直接用让bert attend然后取target出来poolout
  
 
# @Deprecated
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
  def init_hook(self):
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.fl_rate = 0

  def cal_loss(self, out, labels):
    assert len(labels.shape) == 1
    assert len(out.shape) == 2
    assert labels.shape[0] == out.shape[0]
    total = []
    for o, l in zip(out, labels):
      pt = o if (l == 1) else (1 - o)
      loss = (-1) * t.log(pt) * t.pow((1 - pt), self.fl_rate)
      total.append(loss)
    total = t.stack(total)
    return total.sum()

  def pool_policy(self, ss, pos):
    # NOTE: [CLS]
    return B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, pos) # (784)
  
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pools = []
    for ss, pos in zip(sss, poss): 
      pooled = self.pool_policy(ss, pos) # (784)
      pools.append(pooled)
    pools = t.stack(pools) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
    o = self.classifier(pools) # (batch, 1)
    loss = self.cal_loss(o, labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pools = []
    for ss, pos in zip(sss, poss): 
      pooled = self.pool_policy(ss, pos) # (784)
      pools.append(pooled)
    pools = t.stack(pools) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
    o = self.classifier(pools) # (batch, 1)
    self.print_train_info(o, labels, -1)
    return fit_sigmoided_to_label(o), labels 

class Model_Baseline_SEP(Model_Baseline):
  def pool_policy(self, ss, pos):
    return B.compress_by_ss_pos_get_sep(self.bert, self.toker, ss, pos)

class Model_Mean_Pool(Model_Baseline):
  def pool_policy(self, ss, pos):
    embs = B.compress_by_ss_pos_get_emb(self.bert, self.toker, ss, pos) # (?, 784)
    return embs.mean(0) # (784)
  

def fit_sigmoided_to_label(out):
  assert len(out.shape) == 2
  results = []
  for item in out:
    assert item >= 0 and item <= 1
    if item < 0.5:
      results.append(0) 
    else:
      results.append(1) 
  return t.LongTensor(results)


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
  G[f'losses_{index}'] = losses
  dic = {
    'testdic': G[f'testdic_{index}'],
    'devdic': G[f'devdic_{index}'],
    'losses': losses
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

epoch_num = 3

def run_left_right_without_fl():
  init_G(1)
  for i in range(epoch_num):
    G['m'] = m = Model_Fuck()
    m.fl_rate = 0
    get_datas(0, 2, f'左右池化')

def run_left_right_fl():
  init_G(1)
  for i in range(epoch_num):
    G['m'] = m = Model_Fuck()
    m.fl_rate = 5
    get_datas(0, 2, f'左右池化, flrate={m.fl_rate}')

def run_neo_without_fl():
  init_G(1)
  for i in range(epoch_num):
    G['m'] = m = Model_Baseline() # [CLS]
    m.fl_rate = 0
    get_datas(i + 0, 2, f'1:2 [CLS]池化, flrate={m.fl_rate}')
  for i in range(epoch_num):
    G['m'] = m = Model_Baseline_SEP() # [SEP]
    m.fl_rate = 0
    get_datas(i + 10, 2, f'1:2 [SEP]池化, flrate={m.fl_rate}')
  for i in range(epoch_num):
    G['m'] = m = Model_Mean_Pool() # mean
    m.fl_rate = 0
    get_datas(i + 20, 2, f'1:2 [MEAN]池化, flrate={m.fl_rate}')

def run_neo_fl():
  init_G(1)
  for i in range(epoch_num):
    G['m'] = m = Model_Baseline() # [CLS]
    m.fl_rate = 5
    get_datas(i + 30, 2, f'1:2 [CLS]池化, flrate={m.fl_rate}')
  for i in range(epoch_num):
    G['m'] = m = Model_Baseline_SEP() # [SEP]
    m.fl_rate = 5 
    get_datas(i + 40, 2, f'1:2 [SEP]池化, flrate={m.fl_rate}')
  for i in range(epoch_num):
    G['m'] = m = Model_Mean_Pool() # mean
    m.fl_rate = 5
    get_datas(i + 50, 2, f'1:2 [MEAN]池化, flrate={m.fl_rate}')

def run_neo_test():
  init_G(1)
  G['ld'].ds.datas = G['ld'].ds.datas[:15]
  G['testld'].ds.datas = G['testld'].ds.datas[:15]
  G['devld'].ds.datas = G['devld'].ds.datas[:15]
  #G['m'] = m = Model_Baseline() # [CLS]
  #get_datas(0, 1, f'1:2 [cls]池化, flrate={m.fl_rate}, test')
  #G['m'] = m = Model_Baseline_SEP() # [CLS]
  #get_datas(0, 1, f'1:2 [sep]池化, flrate={m.fl_rate}, test')
  #G['m'] = m = Model_Mean_Pool() # [CLS]
  #get_datas(0, 1, f'1:2 [mean]池化, flrate={m.fl_rate}, test')
  G['m'] = m = Model_Fuck() 
  m.fl_rate = 5
  get_datas(0, 1, f'1:2 左右横跳, flrate={m.fl_rate}, test')


def run_neo():
  run_neo_without_fl()
  run_neo_fl()
  run_left_right_without_fl()
  run_left_right_fl()

def greedy_search_flrate():
  init_G(1)
  for i in range(8):
    G['m'] = m = Model_Fuck()
    m.fl_rate = i
    get_datas(i, 2, f'1:2 左右横跳fl rate搜索, flrate={m.fl_rate}')


def run():
  greedy_search_flrate()
