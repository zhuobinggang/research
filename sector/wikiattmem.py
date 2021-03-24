from wikiatt import *
from self_attention import Multihead_SelfAtt, Multihead_Official, Multihead_Official_Scores

class AttMemNet(WikiAttOfficial):
  def init_selfatt_layers(self):
    print(f'Init_selfatt_layers: with head = {self.head}. sentence_integrator will output scores')
    self.sentence_compressor = Multihead_Official_Scores(self.feature, self.head)
    self.sentence_integrator = Multihead_Official_Scores(self.feature, self.head)

  def init_working_memory(self):
    self.working_memory = []
    self.working_memory_info = [] # 用于保存info
    print(f'init_working_memory: memory_size = {self.memory_size}')

  # inpts: [seq_len, (?, feature)], 不定长复数句子
  def cat2memory(self, inpts):
    tensor, info = inpts
    self.working_memory += tensor
    self.working_memory_info += info

  # inpts: [seq_len, (?, feature)], 不定长复数句子
  # return: (seq_len, feature)
  # method: mean pool
  def cls(self, inpts, checking=False):
    results = []
    scores = []
    for inpt in inpts: # (?, feature)
      cls = self.cls_embedding()
      inpt = t.cat([cls, inpt])
      inpt = inpt + self.get_pos_encoding(inpt)# NOTE: pos encoding
      embs, score = self.sentence_compressor(inpt) # (? + 1, feature), (?+1, ?+1)
      cls_pool = embs[0] # (feature)
      results.append(cls_pool) # mean pool
      scores.append(score)
    if checking:
      return t.stack(results), scores
    else:
      return t.stack(results) # (seq_len, feature)

  def memory_self_attention(self):
    # get cls
    cls, word_scores_per_sentence = self.cls(self.working_memory, checking = True)
    # self attend
    integrated, sentence_scores = self.integrate_sentences_info_with_scores(cls)
    return integrated, sentence_scores, word_scores_per_sentence

  def integrate_sentences_info_with_scores(self, cls):
    seq_len, feature = cls.shape
    cls = cls + self.get_pos_encoding(cls)# NOTE: pos encoding
    integrated, scores = self.sentence_integrator(cls) # (seq_len, feature), (seq_len, seq_len)
    return integrated, scores

  def sentence_compressor_scores(self):
    # get cls
    cls, scores = self.cls(self.working_memory, checking = True)
    return scores

  # scores: (seq_len)
  def memory_arrange(self, score):
    seq_len = score.shape[0]
    assert seq_len == len(self.working_memory)
    memory_size = self.memory_size
    cut_point = seq_len - memory_size
    if cut_point < 1:
      pass
    else: 
      _, idx = score.sort() # (seq_len)
      idx_to_pop = idx[:cut_point]
      for idx in idx_to_pop:
        self.working_memory[idx] = None
        self.working_memory_info[idx] = None
      self.working_memory = [m for m in self.working_memory if m is not None]
      self.working_memory_info = [m for m in self.working_memory_info if m is not None]
      # print(len(self.working_memory))
    if self.memory_checking:
      print([round(item, 5) for item in score.tolist()])
    
  # inpts: ss_tensor, ss
  # ss_tensor: [seq_len, (?, feature)], 不定长复数句子
  # ss: [seq_len, string]
  # labels: (label, pos)
  def train(self, inpts, labels, checking = False):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    # cat with working memory
    seq_len = len(inpts)
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    scores = self.adapt_multi_head_scores(sentence_scores) # (seq_len + current_memory_size, seq_len + current_memory_size)
    scores = scores[-seq_len:]
    memory_info_copy = self.working_memory_info.copy() if checking else None
    self.memory_arrange(scores[pos])
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 2)
    loss = self.CEL(o, label)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())
    return loss.detach().item()

  def adapt_multi_head_scores(self, scores):
    shape = scores.shape
    if len(shape) == 4: 
      batch, head, seq_len, _ = shape
      avg_scores = scores.sum(dim=1) / head # (batch, seq_len, seq_len)
      avg_scores = avg_scores.view(seq_len, seq_len) # (seq_len, seq_len)
      return avg_scores
    elif len(shape) == 3:
      batch, seq_len, _ = shape
      scores = scores.view(seq_len, seq_len)
      return scores

  @t.no_grad()
  def dry_run(self, inpts, labels=None, checking = False):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    # cat with working memory
    seq_len = len(inpts)
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    scores = self.adapt_multi_head_scores(sentence_scores) # (seq_len + current_memory_size, seq_len + current_memory_size)
    scores = scores[-seq_len:]
    memory_info_copy = self.working_memory_info.copy() if checking else None
    self.memory_arrange(scores[pos])
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 2)
    self.print_train_info(o, label, -1)
    if checking:
      return o.argmax(1), sentence_scores, word_scores_per_sentence, memory_info_copy
    else:
      return o.argmax(1)


class AttMemNet_Parameter1(AttMemNet):
  def init_selfatt_layers(self):
    print(f'双层sentence compressor')
    self.sentence_compressor = nn.Sequential(
      Multihead_Official(self.feature, self.head), 
      Multihead_Official(self.feature, self.head), 
    )
    self.sentence_integrator = Multihead_Official_Scores(self.feature, self.head)

class AttMemNet_Parameter2(AttMemNet):
  def init_selfatt_layers(self):
    print(f'双层sentence integrator')
    self.sentence_compressor = Multihead_Official(self.feature, self.head)
    self.sentence_integrator = nn.Sequential(
      Multihead_Official(self.feature, self.head), 
      Multihead_Official_Scores(self.feature, self.head)
    )


def run():
  init_G(2)
  head = 6
  size = 5
  for i in range(5):
    G['m'] = m = AttMemNet_Parameter2(hidden_size = 256, head=head, memory_size = size)
    G[f'm1_epoch{i}'] = m
    get_datas(i, 1, f'双层sentence integrator length=1:1 epoch = {i}, head = {head}, size = {size}')

  for i in range(5):
    G['m'] = m = AttMemNet_Parameter1(hidden_size = 256, head=head, memory_size = size)
    G[f'm2_epoch{i}'] = m
    get_datas(i, 1, f'双层sentence compressor length=1:1 epoch = {i}, head = {head}, size = {size}')


