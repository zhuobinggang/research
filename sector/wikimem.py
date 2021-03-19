from wiki import *
from self_attention import Self_Att


def pad(the_list):
  return t.nn.utils.rnn.pad_sequence(the_list, batch_first = True)

class MemNet(WikiSector):
  def init_hook(self):
    self.working_memory = []
    self.working_memory_max_len = 5
    self.memory_checking = False
    self.self_att_output_scores = Self_Att(self.hidden_size) # For working memory
    self.minify_to_cat = nn.Linear(self.hidden_size * 2, self.hidden_size) # For working memory
    self.gru_batch_first_word_compressor = t.nn.GRU(self.wordvec_size, self.hidden_size, batch_first=True)
    self.bi_gru_batch_first_integrator = t.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True)
    self.classifier = nn.Sequential(
      nn.Linear(self.hidden_size * 2, self.hidden_size),
      nn.LeakyReLU(0.1),
      nn.Linear(self.hidden_size, 2),
    )

  def get_should_update(self):
    return chain(self.bi_gru_batch_first_integrator.parameters(), self.classifier.parameters(), self.gru_batch_first_word_compressor.parameters(), self.self_att_output_scores.parameters(), self.minify_to_cat.parameters())

  # item: {'token_id': (max_id_len), 'attend_mark': (max_id_len)}, LongTensor
  def fill_working_memory(self, item):
    self.working_memory.append(item)

  def working_memory_self_att(self):
    mems = pad(self.working_memory) # (seq_len, max_len, feature)
    mems = self.cls(mems) # (seq_len, hidden_size)
    mems, scores = self.self_att_output_scores(mems) # (seq_len, hidden_size)
    return mems, scores

  # score: (seq_len)
  def arrange_working_memory_by_score(self, score):
    if len(self.working_memory) > self.working_memory_max_len: # remove item if out of length
      pos_to_remove = score.argmin().item()
      # print(f'pop: {pos_to_remove}')
      pop_guy = self.working_memory.pop(pos_to_remove)
    else:
      pass  # Do nothing

  # 不该弹出刚刚放进来的
  def arrange_working_memory_by_score_except_last(self, score):
    self.arrange_working_memory_by_score(score[:-1])

  # inpts: (seq_len, words_padded, feature)
  def get_recall_info_then_update_working_memory(self, current_item):
    # Fill working memory with current item
    self.fill_working_memory(current_item) # append
    self_att_mems, scores = self.working_memory_self_att() # (?, hidden_size)
    recall_info = self_att_mems[-1] # (hidden_size)
    score = scores[-1] # (seq_len)
    if self.memory_checking:
      print(score)
    self.arrange_working_memory_by_score_except_last(score) # 如果满了就删除权重最低的
    return recall_info

  # inpts: (seq_len, words_padded, feature)
  def train(self, inpts, labels):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = inpts.cuda()
      label = label.cuda()
    embs = self.cls(inpts) # (seq_len, hidden_size)
    embs = self.integrate_sentences_info(embs) # (seq_len, hidden_size * 2)
    emb = embs[pos] # (hidden_size * 2)
    emb = self.minify_to_cat(emb) # (hidden_size)
    recall_info = self.get_recall_info_then_update_working_memory(inpts[pos]) # (hidden_size)
    o = t.cat([recall_info, emb]) # (hidden_size * 2)
    o = o.view(1, self.hidden_size * 2)
    o = self.classifier(o) # (1, 2)
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
      inpts = inpts.cuda()
      label = label.cuda()
    embs = self.cls(inpts) # (seq_len, hidden_size)
    embs = self.integrate_sentences_info(embs) # (seq_len, hidden_size * 2)
    emb = embs[pos] # (hidden_size * 2)
    emb = self.minify_to_cat(emb) # (hidden_size)
    recall_info = self.get_recall_info_then_update_working_memory(inpts[pos]) # (hidden_size)
    o = t.cat([recall_info, emb]) # (hidden_size * 2)
    o = o.view(1, self.hidden_size * 2)
    o = self.classifier(o) # (1, 2)
    self.print_train_info(o, label, -1)
    return o.argmax(1)


class MemNet_Queue(MemNet):
  # Pop first one as queue
  def arrange_working_memory_by_score(self, _):
    if len(self.working_memory) > self.working_memory_max_len: # remove item if out of length
      pos_to_remove = 0 # NOTE
      pop_guy = self.working_memory.pop(pos_to_remove)
    else:
      pass  # Do nothing

  def arrange_working_memory_by_score_except_last(self, score):
    self.arrange_working_memory_by_score(-1)


# 确认memnet真的有增幅效果
def run():
  init_G(2)
  for i in range(3):
    G['m'] = m = WikiSector(hidden_size = 256)
    get_datas(i, 1, 'dd')
  for i in range(3):
    G['m'] = m = MemNet(hidden_size = 256)
    m.working_memory_max_len = 5
    get_datas(i+10, 1, 'dd')
  for i in range(3):
    G['m'] = m = MemNet_Queue(hidden_size = 256)
    m.working_memory_max_len = 5
    get_datas(i+20, 1, 'dd')
    


