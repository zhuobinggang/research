from wiki import *
from self_attention import Multihead_SelfAtt
import utils_lite as U

# 可变长inpts (seq_len, ?, 300)
class Loader_Var_Len():
  def __init__(self, ds, max_len = 64):
    self.ds = ds
    self.dataset = ds
    self.start = 0
    self.batch_size = self.ds.ss_len
    self.max_len = max_len

  def __iter__(self):
    return self

  def __len__(self):
    return len(self.ds.datas)

  def __next__(self):
    if self.start >= len(self):
      self.start = 0
      raise StopIteration()
    else:
      ss, label, pos_relative = self.ds[self.start]
      self.start += 1
      ss_tensor = w2v(ss, self.max_len) # (seq_len, ?, 300), list
      # ss_padded = t.nn.utils.rnn.pad_sequence(ss_tensor, True) # (seq_len, max_token, 300)
      return ss_tensor, (t.LongTensor([label]), t.LongTensor([pos_relative]))

  def shuffle(self):
    self.ds.shuffle()

def ld(ss_len = 2, max_len = 64):
  ds = DatasetPos(ss_len = ss_len)
  ds.set_datas(D.read_trains())
  return Loader_Var_Len(ds, max_len)

def tld(ss_len = 2, max_len = 64):
  ds = DatasetPos(ss_len = ss_len)
  ds.set_datas(D.read_tests())
  return Loader_Var_Len(ds, max_len)

def dld(ss_len = 2, max_len = 64):
  ds = DatasetPos(ss_len = ss_len)
  ds.set_datas(D.read_devs())
  return Loader_Var_Len(ds, max_len)

def init_G(length):
  G['ld'] = ld(ss_len = length, max_len = 64)
  G['testld'] = tld(ss_len = length, max_len = 64)
  G['devld'] = dld(ss_len = length, max_len = 64)


class WikiAtt(WikiSector):
  def init_hook(self):
    self.feature = 300
    self.classifier = nn.Sequential(
      nn.Linear(self.feature, int(self.feature * 1.5)), 
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.feature * 1.5), int(self.feature / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.feature / 2), 2),
    )
    self.sentence_compressor = nn.Sequential(
      Multihead_SelfAtt(self.feature, 4),
    )
    self.sentence_integrator = nn.Sequential(
      Multihead_SelfAtt(self.feature, 4),
    )
    self.ember = nn.Embedding(3, self.feature)
    # self.pos_embedding = nn.Embedding(20, self.feature)

  def get_should_update(self):
    return chain(self.classifier.parameters(), self.sentence_compressor.parameters(), self.sentence_integrator.parameters(), self.ember.parameters())

  # ss: (seq_len, feature)
  # return: (seq_len, feature)
  def integrate_sentences_info(self, ss):
    seq_len, feature = ss.shape
    pos = U.position_encoding(ss) # NOTE: pos encoding
    ss = (ss + pos).float()
    integrated = self.sentence_integrator(ss) # (seq_len, feature), (seq_len, seq_len)
    return integrated

  def cls_embedding(self):
    idx = t.LongTensor([0])
    return self.ember(idx.cuda() if GPU_OK else idx)

  # inpts: [seq_len, (?, feature)], 不定长复数句子
  # return: (seq_len, feature)
  # method: mean pool
  def cls(self, inpts):
    results = []
    for inpt in inpts: # (?, feature)
      cls = self.cls_embedding()
      inpt = t.cat([cls, inpt])
      pos = U.position_encoding(inpt) # NOTE: pos encoding
      inpt = (inpt + pos).float()
      embs = self.sentence_compressor(inpt) # (? + 1, feature), (?+1, ?+1)
      cls_pool = embs[0] # (feature)
      results.append(cls_pool) # mean pool
    return t.stack(results) # (seq_len, feature)

  # inpts: [seq_len, (?, feature)], 不定长复数句子
  # return: (seq_len, feature)
  # method: mean pool
  def cls_mean_pool(self, inpts):
    results = []
    for inpt in inpts: # (?, feature)
      embs = self.sentence_compressor(inpt) # (? + 1, feature), (?+1, ?+1)
      results.append(embs.mean(0)) # mean pool
    return t.stack(results) # (seq_len, feature)

  def learning_rate(self):
    return 1e-3

  # inpts: [seq_len, (?, feature)], 不定长复数句子
  # labels: (label, pos)
  def train(self, inpts, labels):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    embs = self.cls(inpts) # (seq_len, feature)
    embs = self.integrate_sentences_info(embs) # (seq_len, hidden_size * 2)
    emb = embs[pos] # (feature)
    emb = emb.view(1, self.feature)
    o = self.classifier(emb) # (1, 2)
    loss = self.CEL(o, label)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    embs = self.cls(inpts) # (seq_len, feature)
    embs = self.integrate_sentences_info(embs) # (seq_len, hidden_size * 2)
    emb = embs[pos] # (feature)
    emb = emb.view(1, self.feature)
    o = self.classifier(emb) # (1, 2)
    self.print_train_info(o, label, -1)
    return o.argmax(1)


# 确认两边长度1,2,3对结果的影响
def run():
  init_G(2)
  G['m'] = m = WikiAtt(hidden_size = 256)
  get_datas(0, 1, 'wiki2vec, 1:1')
  print(R.G)
