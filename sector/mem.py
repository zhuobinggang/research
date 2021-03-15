from exp4 import *


class Self_Att(nn.Module):
  def __init__(self, feature):
    super().__init__()
    self.feature = feature
    self.squre_dk = np.sqrt(self.feature)
    self.WQ = nn.Linear(feature, feature)
    self.WK = nn.Linear(feature, feature)
    self.WV = nn.Linear(feature, feature)

  # embs: (seq_len, feature)
  # return: (seq_len, feature)
  def forward(self, embs):
    qs = self.WQ(embs) # (seq_len, feature)
    ks = self.WK(embs) # (seq_len, feature)
    vs = self.WV(embs) # (seq_len, feature)
    scores = t.mm(qs, ks.transpose(0, 1)) # (seq_len, seq_len)
    scores = t.softmax(scores / self.squre_dk, 1) # (seq_len, seq_len)
    result = t.mm(scores, vs) # (seq_len, feature)
    return result, scores.detach()


# 带有带POS的working memory
class Model_Mem(BERT_LONG_TF_POS):
  def init_hook(self):
    self.self_att_layer = nn.TransformerEncoderLayer(d_model=self.bert_size, nhead=8, dim_feedforward=int(self.bert_size * 1.5), dropout=0)
    self.self_att_output_scores = Self_Att(self.bert_size)
    self.working_memory = []
    self.working_memory_max_len = 5
    self.classifier = nn.Sequential( # (1, 768) => (1, 2)
      nn.Linear(self.bert_size, 2),
    )
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 3]))
    self.CEL = nn.CrossEntropyLoss()
    # self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, 4])) # LSTM比较难训练，试着

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.self_att_layer.parameters(), self.self_att_output_scores.parameters())

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

  # item: {'token_id': (max_id_len), 'attend_mark': (max_id_len)}, LongTensor
  def fill_working_memory(self, item):
    self.working_memory.append(item)

  def working_memory_self_att(self):
    token_ids = []
    attend_marks = []
    for item in self.working_memory:
      token_ids.append(item.token_id)
      attend_marks.append(item.attend_mark)
    token_ids = t.stack(token_ids) # (seq_len, max_id_len)
    attend_marks = t.stack(attend_marks) # (seq_len, max_id_len)
    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (seq_len, 768)
    embs = self.processed_embs(embs) # (seq_len, 768)
    embs, scores = self.self_att_output_scores(embs)
    return embs, scores

  # score: (seq_len)
  def arrange_working_memory_by_score(self, score):
    if len(self.working_memory) > self.working_memory_max_len: # remove item if out of length
      pos_to_remove = score.argmin().item()
      _ = self.working_memory.pop(pos_to_remove)
    else:
      pass  # Do nothing

  # inpts: token_ids, attend_marks
  # token_ids: (sentence_size, max_id_len)
  # labels: (sentence_size), zero/one
  def train(self, inpts, labels):
    token_ids, attend_marks = inpts # token_ids = attend_marks: (sentence_size, max_id_len)
    label, pos = labels # LongTensor([label/pos])
    pos = pos.item()
    if GPU_OK:
      token_ids = token_ids.cuda()
      attend_marks = attend_marks.cuda()
      label = label.cuda()

    item = {'token_id': token_ids[pos], 'attend_mark': attend_marks[pos]}
    # Fill working memory with current item
    self.fill_working_memory(item) 
    recall_infos, scores = self.working_memory_self_att()
    recall_info = recall_infos[-1] # (768)
    score = scores[-1] # (seq_len)
    self.arrange_working_memory_by_score(score) # 如果满了就删除权重最低的

    embs = self.get_batch_cls_emb(token_ids, attend_marks) # (sentence_size, 768)
    embs = self.processed_embs(embs) # (sentence_size, 768)

    now_info = self.integrate_sentences_info(embs) # (sentence_size, 768)
    now_info = now_info[pos] # (768)

    
    o = t.cat([recall_info, now_info]) # (768 * 2)

    o = o.view(1, self.bert_size * 2) # (1, 768)
    o = self.classifier(o) # (1, 2)
    loss = self.CEL(o, label)

    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())


    return loss.detach().item()
