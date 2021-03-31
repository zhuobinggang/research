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
Loader_Symmetry = data.Loader_Symmetry

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
  def __init__(self, weight_one = 1, hidden_size = 256, head = 8, dropout=0, rate = 0):
    super().__init__()
    self.fl_rate = rate
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

  def cal_loss(self, out, labels, rate = None):
    assert len(labels.shape) == 1
    assert len(out.shape) == 2
    assert labels.shape[0] == out.shape[0]
    rate = self.fl_rate if rate is None else rate
    total = []
    for o, l in zip(out, labels):
      pt = o if (l == 1) else (1 - o)
      loss = (-1) * t.log(pt) * t.pow((1 - pt), rate)
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


class Model_Sector_Plus(Model_Fuck):
  def init_hook(self): 
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.classifier2 = nn.Sequential(
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.classifier2.parameters())
  
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    clss_for_order_checking = [] # For order detector
    order_labels = [] # For order detector
    for ss, pos in zip(sss, poss):
      left, right = get_left_right_by_ss_pos(ss, pos)
      emb1 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb2 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      # print(f'{emb1.shape[0]}, {emb2.shape[0]}')
      assert emb1.shape[0] == emb2.shape[0]
      mean = (emb1 + emb2) / 2 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
      # For order detector
      if len(left) == 2: 
        if random.randrange(100) > 50: # 1/2的概率倒序
          left_disturbed = list(reversed(left))
          order_labels.append(1)
        else:
          left_disturbed = left.copy()
          order_labels.append(0)
        cls = B.compress_by_ss_pos_get_cls(self.bert, self.toker, left_disturbed, 1) # (784)
        clss_for_order_checking.append(cls)
      else:
        print(f'Warning, left length = {len(left)}')
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    order_labels = t.LongTensor(order_labels) # (x <= batch) For order detector
    if GPU_OK:
      labels = labels.cuda()
      order_labels = order_labels.cuda()
    o = self.classifier(pooled_embs) # (batch, 1)
    sector_loss = self.cal_loss(o, labels)
    # For order detector
    clss_for_order_checking = t.stack(clss_for_order_checking) # (x <= batch, 784)
    output_ordering = self.classifier2(clss_for_order_checking) # (x, 1)
    ordering_loss = self.cal_loss(output_ordering, order_labels, rate=0) # 不存在数据不均衡问题
    loss = sector_loss + ordering_loss
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

# 不要分开CLS和mean，同一个embedding需要考虑两样东西
# C: 如果它利用位置信息就很容易区分了
class Model_Sector_Plus_V2(Model_Sector_Plus):
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    clss_for_order_checking = [] # For order detector
    order_labels = [] # For order detector
    for ss, pos in zip(sss, poss):
      left, right = get_left_right_by_ss_pos(ss, pos)
      emb1 = B.compress_left_get_embs(self.bert, self.toker, left) # (seq_len, 784)
      emb2 = B.compress_right_get_embs(self.bert, self.toker, right) # (seq_len, 784)
      # print(f'{emb1.shape[0]}, {emb2.shape[0]}')
      assert emb1.shape[0] == emb2.shape[0]
      mean = (emb1 + emb2) / 2 # (seq_len, 784)
      pooled = mean.mean(0) # (784)
      pooled_embs.append(pooled)
      # For order detector
      if len(left) == 2: 
        if random.randrange(100) > 50: # 1/2的概率倒序
          left_disturbed = list(reversed(left))
          order_labels.append(1)
        else:
          left_disturbed = left.copy()
          order_labels.append(0)
        cls = B.compress_by_ss_pair_get_mean(self.bert, self.toker, left_disturbed) # (784)
        clss_for_order_checking.append(cls)
      else:
        print(f'Warning, left length = {len(left)}')
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    order_labels = t.LongTensor(order_labels) # (x <= batch) For order detector
    if GPU_OK:
      labels = labels.cuda()
      order_labels = order_labels.cuda()
    o = self.classifier(pooled_embs) # (batch, 1)
    sector_loss = self.cal_loss(o, labels)
    # For order detector
    clss_for_order_checking = t.stack(clss_for_order_checking) # (x <= batch, 784)
    output_ordering = self.classifier2(clss_for_order_checking) # (x, 1)
    ordering_loss = self.cal_loss(output_ordering, order_labels, rate=0) # 不存在数据不均衡问题
    loss = sector_loss + ordering_loss
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()


# By [CLS]
class Double_Sentence_CLS(Model_Fuck):
  def pool_policy(self, ss, pos):
    # NOTE: [CLS]
    return B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, pos) # (784)

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    for ss, pos in zip(sss, poss):
      if pos != 2:
        print(f'Warning: pos={pos}')
      emb = self.pool_policy(ss, pos)
      pooled_embs.append(emb)
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
      if pos != 2:
        print(f'Warning: pos={pos}')
      emb = self.pool_policy(ss, pos)
      pooled_embs.append(emb)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    if GPU_OK:
      labels = labels.cuda()
    o = self.classifier(pooled_embs) # (batch, 1)
    self.print_train_info(o, labels, -1)
    return fit_sigmoided_to_label(o), labels

class Double_Sentence_MEAN(Double_Sentence_CLS):
   def pool_policy(self, ss, pos):
    # NOTE: [MEAN]
    return B.compress_by_ss_pair_get_mean(self.bert, self.toker, ss, pos) # (784) 

class Double_Sentence_Plus_Ordering(Double_Sentence_CLS):
  def init_hook(self): 
    self.classifier = nn.Sequential(
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.classifier2 = nn.Sequential(
      nn.Linear(self.bert_size, 1),
      nn.Sigmoid()
    )
    self.ordering_loss_rate = 0.5

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.classifier2.parameters())

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    pooled_embs = [] 
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      if pos != 2:
        print(f'Warning: pos={pos}')
      pooled_embs.append(self.pool_policy(ss, pos))
      # Ordering
      ss_disturbed = ss.copy()
      if random.randrange(100) > 50: # 1/2的概率倒序
        random.shuffle(ss_disturbed)
      else:
        pass
      ordering_embs.append(self.pool_policy(ss_disturbed, pos))
      if ss_disturbed == ss:
        ordering_labels.append(0)
      else:
        ordering_labels.append(1)
    pooled_embs = t.stack(pooled_embs) # (batch, 784)
    ordering_embs = t.stack(ordering_embs)
    labels = t.LongTensor(labels) # (batch), (0 or 1)
    ordering_labels = t.LongTensor(ordering_labels)
    if GPU_OK:
      labels = labels.cuda()
      ordering_labels = ordering_labels.cuda()
    o = self.classifier(pooled_embs) # (batch, 1)
    o_ordering = self.classifier2(ordering_embs) # (batch, 1)
    loss_sector = self.cal_loss(o, labels)
    loss_ordering = self.cal_loss(o_ordering, ordering_labels)
    loss = loss_sector + self.ordering_loss_rate * loss_ordering
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

# ========================================================
  

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

def init_G_Symmetry(half = 1):
  G['ld'] = Loader_Symmetry(ds = data.train_dataset(ss_len = half * 2, max_ids = -1), half = half, batch = 2)
  G['testld'] = Loader_Symmetry(ds = data.test_dataset(ss_len = half * 2, max_ids = -1), half = half, batch = 2)
  G['devld'] = Loader_Symmetry(ds = data.dev_dataset(ss_len = half * 2, max_ids = -1), half = half, batch = 2)

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
  for i in range(7):
    G['m'] = m = Model_Fuck()
    m.fl_rate = i + 1
    get_datas(i, 2, f'1:2 左右横跳fl rate搜索, flrate={m.fl_rate}')

# 左右横跳 rate = 3获取实验结果
def left_right_flrate3_run():
  init_G(1)
  for i in range(5):
    G['m'] = m = Model_Fuck(rate=3)
    get_datas(i, 2, f'1:2 左右横跳fl rate搜索, flrate={m.fl_rate}')

def run_sector_plus_ordering():
  init_G(1)
  G['m'] = m = Model_Sector_Plus(rate = 0)
  get_datas(1, 1, f'1:2 左右横跳 + ordering, flrate={m.fl_rate}, 1')
  get_datas(2, 1, f'1:2 左右横跳 + ordering, flrate={m.fl_rate}, 2')

def ordering_with_cls():
  init_G(1)
  for i in range(3):
    G['m'] = m = Model_Sector_Plus(rate = 0)
    get_datas(i + 10, 2, f'1:2 左右横跳 + [CLS] ordering, flrate={m.fl_rate}')

def ordering_with_mean():
  init_G(1)
  for i in range(3):
    G['m'] = m = Model_Sector_Plus_V2(rate = 0)
    get_datas(i + 20, 2, f'1:2 左右横跳 + [MEAN] ordering, flrate={m.fl_rate}')

def run_double_sentence_exp():
  init_G_Symmetry(2) 
  for i in range(5):
    G['m'] = m = Double_Sentence_Plus_Ordering(rate = 0)
    m.ordering_loss_rate = 1
    get_datas(i, 2, f'2:2 Double_Sentence_Plus_Ordering, flrate={m.fl_rate},ordering_loss_rate= {m.ordering_loss_rate}')
  for i in range(3):
    G['m'] = m = Double_Sentence_Plus_Ordering(rate = 0)
    m.ordering_loss_rate = 0.5
    get_datas(i + 10, 2, f'2:2 Double_Sentence_Plus_Ordering, flrate={m.fl_rate},ordering_loss_rate= {m.ordering_loss_rate}')
  for i in range(3):
    G['m'] = m = Double_Sentence_Plus_Ordering(rate = 0)
    m.ordering_loss_rate = 0.3
    get_datas(i + 20, 2, f'2:2 Double_Sentence_Plus_Ordering, flrate={m.fl_rate},ordering_loss_rate= {m.ordering_loss_rate}')

def run_v1():
  left_right_flrate3_run()
  ordering_with_cls()
  ordering_with_mean()

def run():
  run_double_sentence_exp()

