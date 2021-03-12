from csg_exp3 import *

class BERT_LONG_TF(BERT_LONG_DEPEND):
  def init_hook(self):
    self.self_att_layer = nn.TransformerEncoderLayer(d_model=self.bert_size, nhead=1, dim_feedforward=int(self.bert_size * 1.5), dropout=0.1)
    self.classifier = nn.Sequential( # (1, 768) => (1, 2)
      nn.Linear(self.bert_size, 2),
    )
    self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 3]))
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

# =============== 

def init_G():
  G['ld'] = ld = Loader_Long_Depend(Train_DS_Long_Depend(ss_len = length))
  G['testld'] = testld = Loader_Long_Depend(Test_DS_Long_Depend(ss_len = length))
  G['devld'] = devld = Loader_Long_Depend(Dev_DS_Long_Depend(ss_len = length))
  m = G['m'] = BERT_LONG_TF()
  return m, ld, testld, devld

def train_then_record(m, ld, testld, name, epoch = 1):
  losses = runner.train_simple(m, ld, epoch) # only one epoch for order matter model
  G[name] = runner.get_test_result_long(m, testld)

def get_datas_long_tf(length = 4, index = 0):
  m, ld, testld, devld = init_G()
  train_then_record(m, ld, testld,name=f'testdic_{index}', epoch=1)
  train_then_record(m, ld, devld,name=f'devdic_{index}', epoch=1)
  return losses
