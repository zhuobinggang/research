from csg_exp3 import *
torch = t
import requests
import utils_lite
R = runner
from self_attention import Multihead_SelfAtt, Multihead_Official, Multihead_Official_Scores

def request_my_logger(dic, desc = 'No describe'):
  try:
    url = "https://hookb.in/b9xlr2GnnjC3DDogQ0jY"
    dic['desc'] = desc
    requests.post(url, json=dic)
  except:
    print('Something went wrong in request_my_logger()')

def cuda(emb):
  return emb.cuda() if GPU_OK else emb

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
  return cuda(embs)


class BERT_LONG_TF(BERT_LONG_DEPEND):
  def init_hook(self):
    self.self_att_layer = nn.TransformerEncoderLayer(d_model=self.bert_size, nhead=self.head, dim_feedforward=int(self.bert_size * 1.5), dropout=self.dropout)
    self.classifier = nn.Sequential( # (1, 768) => (1, 2)
      nn.Linear(self.bert_size, 2),
    )
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 3]))
    self.CEL = nn.CrossEntropyLoss()
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 4])) # LSTM比较难训练，试着

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.self_att_layer.parameters())

  # ss: (sentence_size, 768)
  # return: (sentence_size, 768)
  def integrate_sentences_info(self, ss):
    seq_len, feature = ss.shape
    ss = ss.view(seq_len, 1, feature) # (sentence_size, 1, 768)
    ss = self.self_att_layer(ss) # (sentence_size, 1, 768)
    return ss.view(seq_len, feature)

  def cal_loss(self, out, label):
    assert len(label.shape) == 1
    assert len(out.shape) == 2
    assert (out.shape[0] == 1 and out.shape[1] == 2)
    loss = self.CEL(o, label)
    return loss

  # inpts: token_ids, attend_marks
  # token_ids: (sentence_size, max_id_len)
  # labels: (sentence_size), zero/one
  def train(self, inpts, labels):
    token_ids, attend_marks = inpts # token_ids = attend_marks: (sentence_size, max_id_len)
    # labels = self.preprocess_labels(labels)
    label, pos = labels # LongTensor([label/pos])
    pos = pos.item()
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      label = label.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (sentence_size, 768)
    embs = self.processed_embs(embs) # (sentence_size, 768)

    o = self.integrate_sentences_info(embs) # (sentence_size, 768)
    o = o[pos] # (768)
    o = o.view(1, self.bert_size) # (1, 768)
    o = self.classifier(o) # (1, 2)
    loss = self.cal_loss(o, label)

    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())

    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    token_ids, attend_marks = inpts # token_ids = attend_marks: (sentence_size, max_id_len)
    # labels = self.preprocess_labels(labels)
    label, pos = labels # LongTensor([label/pos])
    pos = pos.item()
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      label = label.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (sentence_size, 768)
    embs = self.processed_embs(embs) # (sentence_size, 768)

    o = self.integrate_sentences_info(embs) # (sentence_size, 768)
    o = o[pos] # (768)
    o = o.view(1, self.bert_size) # (1, 768)
    o = self.classifier(o) # (1, 2)

    self.print_train_info(o, label, -1)
    return o.view(-1).argmax().item()



class BERT_LONG_TF_POS(BERT_LONG_TF):
  def init_hook(self):
    self.feature = self.bert_size
    self.self_att_layer = Multihead_Official(feature = self.feature, head = self.head)
    print(f'Init BERT_LONG_TF_POS with head = {self.head}')
    self.classifier = nn.Sequential( # (1, 768) => (1, 2)
      nn.Linear(self.bert_size, 2),
    )
    self.CEL = nn.CrossEntropyLoss()
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 4])) # LSTM比较难训练，试着
    self.pos_matrix = utils_lite.position_matrix(99, self.feature).float()

  def get_pos_encoding(self, emb):
    seq_len, feature = emb.shape
    assert feature == self.feature
    return self.pos_matrix[:seq_len].detach()

  # ss: (sentence_size, 768)
  # return: (sentence_size, 768)
  def integrate_sentences_info(self, ss):
    seq_len, feature = ss.shape
    ss = ss + self.get_pos_encoding(ss) # (seq_len, feature) NOTE: Add pos
    ss = self.self_att_layer(ss) # (seq_len, feature)
    return ss

class BERT_LONG_TF_POS_FL(BERT_LONG_TF_POS):
  def init_hook(self):
    self.fl_rate = 5
    self.feature = self.bert_size
    self.self_att_layer = Multihead_Official(feature = self.feature, head = self.head)
    print(f'Init BERT_LONG_TF_POS with head = {self.head}')
    self.classifier = nn.Sequential( # (1, 768) => (1, 2)
      nn.Linear(self.bert_size, 1), # NOTE: 只输出一个概率
      nn.Sigmoid()
    )
    self.CEL = nn.CrossEntropyLoss()
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 4])) # LSTM比较难训练，试着
    self.pos_matrix = utils_lite.position_matrix(99, self.feature).float()

  def cal_loss(self, out, label):
    assert len(label.shape) == 1
    assert len(out.shape) == 2
    assert (out.shape[0] == 1 and out.shape[1] == 1)
    # loss = self.CEL(o, label)
    pt = out if (label == 1) else (1 - out)
    loss = (-1) * t.log(pt) * t.pow((1 - pt), self.fl_rate)
    return loss

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    token_ids, attend_marks = inpts # token_ids = attend_marks: (sentence_size, max_id_len)
    # labels = self.preprocess_labels(labels)
    label, pos = labels # LongTensor([label/pos])
    pos = pos.item()
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      label = label.cuda()
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (sentence_size, 768)
    embs = self.processed_embs(embs) # (sentence_size, 768)

    o = self.integrate_sentences_info(embs) # (sentence_size, 768)
    o = o[pos] # (768)
    o = o.view(1, self.bert_size) # (1, 768)
    o = self.classifier(o) # (1, 1)
    self.print_train_info(o, label, -1)
    result = 0 if o < 0.5 else 1
    return result

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      result = 0 if o < 0.5 else 1
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {result} Loss: {loss} ')
  


# =============== 

def init_G(length):
  G['ld'] = Loader_Long_Depend(Train_DS_Long_Depend(ss_len = length))
  G['testld'] = Loader_Long_Depend(Test_DS_Long_Depend(ss_len = length))
  G['devld'] = Loader_Long_Depend(Dev_DS_Long_Depend(ss_len = length))

def set_G(m, ld, testld, devld):
  G['ld'] = ld
  G['testld'] = testld
  G['devld'] = devld
  G['m'] = m

def read_G():
  return G['m'], G['ld'], G['testld'], G['devld']

def get_datas(index, epoch, desc):
  m, ld , testld, devld = read_G()
  losses = R.get_datas(m, ld, testld, devld, index, epoch, desc)
  return losses

def run_at_night_15():
  init_G(2)
  base = 10
  for i in range(3):
   G['m'] = m = BERT_LONG_TF_POS_FL(head=8)
   m.fl_rate = 5
   get_datas(i + base, 2, f'BASELINE rate = {m.fl_rate}')

  init_G(4)
  base = 20
  for i in range(3):
   G['m'] = m = BERT_LONG_TF_POS_FL(head=8)
   m.fl_rate = 5
   base = 0
   get_datas(i + base, 2, f'加大长度2:2 rate = {m.fl_rate}')

  init_G(2)
  base = 30
  for i in range(3):
   G['m'] = m = BERT_LONG_TF_POS_FL(head=8)
   m.fl_rate = 6 
   get_datas(i + base, 2, f'1:1但是提升衰减率rate = {m.fl_rate}')

  init_G(6)
  base = 40
  for i in range(3):
   G['m'] = m = BERT_LONG_TF_POS_FL(head=8)
   m.fl_rate = 5
   get_datas(i + base, 2, f'加大长度3:3 rate = {m.fl_rate}')
