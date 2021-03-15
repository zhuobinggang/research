from csg_exp3 import *
torch = t


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
    loss = self.CEL(o, label)

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
  # ss: (sentence_size, 768)
  # return: (sentence_size, 768)
  def integrate_sentences_info(self, ss):
    seq_len, feature = ss.shape
    # NOTE: Add pos
    pos = position_encoding(ss) # (seq_len, feature)
    ss = (ss + pos).float()
    ss = ss.view(seq_len, 1, feature) # (sentence_size, 1, 768)
    ss = self.self_att_layer(ss) # (sentence_size, 1, 768)
    return ss.view(seq_len, feature)


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

def get_datas(index, epoch):
  m, ld, testld, devld = read_G()
  losses = runner.train_simple(m, ld, epoch) # only one epoch for order matter model
  G[f'testdic_{index}'] = runner.get_test_result_long(m, testld)
  G[f'devdic_{index}'] = runner.get_test_result_long(m, devld)
  print(G[f'testdic_{index}'])
  return losses


# length = 3:3, weight = 1:1, head = 4
# length = 3:3, weight = 1:1, head = 16
# length = 3:3, weight = 1:1, head = 24
# length = 3:3, weight = 1:1, head = 32
# length = 3:3, weight = 1:1, head = 8, dropout = 0.1
# length = 4:4, weight = 1:1, head = 8
# length = 5:5, weight = 1:1, head = 8
# length = 6:6, weight = 1:1, head = 8
# 在len=6时候: 1) 测试pos vs no pos的性能差异；2) 对比head=8时候和head=1时候的性能差异
def run_at_night_15():
   init_G(6)

   # length = 3:3, weight = 1:1, head = 4
   G['m'] = m = BERT_LONG_TF_POS(head=4)
   get_datas(0, 2)

   # length = 3:3, weight = 1:1, head = 8
   G['m'] = m = BERT_LONG_TF_POS(head=8)
   get_datas(1, 2)

   # length = 3:3, weight = 1:1, head = 16
   G['m'] = m = BERT_LONG_TF_POS(head=16)
   get_datas(2, 2)

   # length = 3:3, weight = 1:1, head = 24
   G['m'] = m = BERT_LONG_TF_POS(head=24)
   get_datas(3, 2)

   # length = 3:3, weight = 1:1, head = 8, dropout = 0.1
   G['m'] = m = BERT_LONG_TF_POS(head=8, dropout=0.1)
   get_datas(4, 2)



