from wikiatt import *
from self_attention import Multihead_SelfAtt, Multihead_Official, Multihead_Official_Scores
import random

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

  def cal_loss(self, out, label):
    assert len(label.shape) == 1
    assert len(out.shape) == 2
    assert (out.shape[0] == 1 and out.shape[1] == 2)
    loss = self.CEL(o, label)
    return loss
    
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
    seq_len = len(inpts[0])
    current_mem_len = len(self.working_memory)
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    assert out.shape[0] == current_mem_len + seq_len
    out = out[-seq_len:] # 剪掉记忆储存部分
    scores = self.adapt_multi_head_scores(sentence_scores) # (seq_len + current_memory_size, seq_len + current_memory_size)
    assert scores.shape[0] == scores.shape[1] == current_mem_len + seq_len 
    scores = scores[-seq_len:]
    memory_info_copy = self.working_memory_info.copy() if checking else None
    self.memory_arrange(scores[pos])
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 2)
    loss = self.cal_loss(o, label)
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


class AttMemNet_FL(AttMemNet):
  def init_hook(self):
    self.feature = 300
    self.fl_rate = 5
    self.max_seq_len = 64
    self.classifier = nn.Sequential(
      nn.Linear(self.feature, int(self.feature / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.feature / 2), 1),
      nn.Sigmoid()
    )
    self.init_selfatt_layers()
    self.init_working_memory()
    self.ember = nn.Embedding(3, self.feature)
    self.pos_matrix = U.position_matrix(self.max_seq_len + 10, self.feature).float()

  def cal_loss(self, out, label):
    assert len(label.shape) == 1
    assert len(out.shape) == 2
    assert (out.shape[0] == 1 and out.shape[1] == 1)
    # loss = self.CEL(o, label)
    pt = out if (label == 1) else (1 - out)
    loss = (-1) * t.log(pt) * t.pow((1 - pt), self.fl_rate)
    return loss

  @t.no_grad()
  def dry_run(self, inpts, labels=None, checking = False):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    # cat with working memory
    seq_len = len(inpts[0])
    current_mem_len = len(self.working_memory)
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    assert out.shape[0] == current_mem_len + seq_len
    out = out[-seq_len:] # 剪掉记忆储存部分
    scores = self.adapt_multi_head_scores(sentence_scores) # (seq_len + current_memory_size, seq_len + current_memory_size)
    assert scores.shape[0] == scores.shape[1] == current_mem_len + seq_len 
    scores = scores[-seq_len:]
    memory_info_copy = self.working_memory_info.copy() if checking else None
    self.memory_arrange(scores[pos])
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 1)
    probability = o.item()
    self.print_train_info(o, label, -1)
    result = 0 if o < 0.5 else 1
    if checking:
      return result, sentence_scores, word_scores_per_sentence, memory_info_copy, probability
    else:
      return result

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      result = 0 if o < 0.5 else 1
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {result} Loss: {loss} ')


# 0 memory size, 不保留奇怪的内存管理规则
class Att_FL(AttMemNet_FL):
  def empty_memory(self):
    self.working_memory = []
    self.working_memory_info = []

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
    seq_len = len(inpts[0])
    self.empty_memory()
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    memory_info_copy = self.working_memory_info.copy() if checking else None
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 1)
    loss = self.cal_loss(o, label)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, inpts, labels=None, checking = False):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    # cat with working memory
    seq_len = len(inpts[0])
    self.empty_memory()
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    memory_info_copy = self.working_memory_info.copy() if checking else None
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 1)
    probability = o.item()
    self.print_train_info(o, label, -1)
    result = 0 if o < 0.5 else 1
    if checking:
      return result, sentence_scores, word_scores_per_sentence, memory_info_copy, probability
    else:
      return result

  def print_train_info(self, o, labels=None, loss=-1):
    if self.verbose:
      result = 0 if o < 0.5 else 1
      if labels is None:
        labels = t.LongTensor([-1])
      print(f'Want: {labels.tolist()} Got: {result} Loss: {loss} ')
 

class Att_FL_Ordering(Att_FL):
  def init_hook(self):
    self.feature = 300
    self.fl_rate = 0
    self.max_seq_len = 64
    self.classifier = nn.Sequential(
      nn.Linear(self.feature, int(self.feature / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.feature / 2), 1),
      nn.Sigmoid()
    )
    # NOTE: 增加classfier2
    self.classifier2 = nn.Sequential(
      nn.Linear(self.feature, int(self.feature / 2)),
      nn.LeakyReLU(0.1),
      nn.Linear(int(self.feature / 2), 1),
      nn.Sigmoid()
    )
    self.init_selfatt_layers()
    self.init_working_memory()
    self.ember = nn.Embedding(3, self.feature)
    self.pos_matrix = U.position_matrix(self.max_seq_len + 10, self.feature).float()

  def get_should_update(self):
    # NOTE: 增加classfier2
    return chain(self.classifier.parameters(), self.sentence_compressor.parameters(), self.sentence_integrator.parameters(), self.ember.parameters(), self.classifier2.parameters())

  def get_ordering_loss(self, ss, pos):
    ss_disturbed = ss.copy() # 
    ss_disturbed = [''.join(words) for words in ss_disturbed]
    if random.randrange(100) > 50: # 1/2的概率倒序
      random.shuffle(ss_disturbed)
      label_ordering = t.LongTensor([0 if ss_disturbed == ss else 1])
    else:
      label_ordering = t.LongTensor([0])
    ss_disturbed_tensor, sentence_words = w2v(ss_disturbed, 64, require_words = True) # (seq_len, ?, feature)
    self.empty_memory()
    self.cat2memory((ss_disturbed_tensor, sentence_words))
    out, _, _ = self.memory_self_attention() # (seq_len + current_memory_size, feature)
    out = out[-seq_len:] # 剪掉记忆储存部分 (seq_len, feature)
    out = out[pos] # (feature)
    out = self.classfier2(out) # (1)
    out.view(1, 1)
    return self.cal_loss(out, label_ordering)

  # inpts: ss_tensor, ss
  # ss_tensor: [seq_len, (?, feature)], 不定长复数句子
  # ss: [seq_len, string]
  # labels: (label, pos)
  def train(self, inpts, labels, checking = False):
    label, pos = labels # (1), LongTensor
    ss_tensor, ss = inpts
    pos = pos.item()
    if GPU_OK:
      ss_tensor = [item.cuda() for item in ss_tensor]
      label = label.cuda()
    # cat with working memory
    seq_len = len(ss)
    self.empty_memory()
    self.cat2memory(inpts)
    out, sentence_scores, word_scores_per_sentence = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    memory_info_copy = self.working_memory_info.copy() if checking else None
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 1)
    loss_sector = self.cal_loss(o, label)
    loss_ordering = self.get_ordering_loss(ss, pos)
    loss = loss_sector + loss_ordering
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, label, loss.detach().item())
    return loss.detach().item()

# ===========================

def run():
  init_G(2)
  head = 6
  memsize = 0
  G['m'] = m = Att_FL_Ordering(hidden_size = 256, head=head, memory_size = memsize)
  m.fl_rate = 0
  epochs = 1
  get_datas(0, epochs, f'Ordering, length=2:2 epochs = {epochs}, head = {head}, size = {memsize}, fl_rate = {m.fl_rate}')


def get_analysis_data(m):
  G['results'] = results = []
  devld = G['devld']
  devld.start = 0
  for inpts, labels in devld:
    _, sentence_scores, word_scores_per_sentence, memory_info_copy, probability = m.dry_run(inpts, labels, checking = True)
    label = labels[0].item()
    results.append((label, probability, memory_info_copy, sentence_scores, word_scores_per_sentence))


def filter_datas_1(results):
  return [res for res in results if res[0] == 1]

def most_near_1(results):
  return list(reversed(sorted(results, key = lambda x: x[1]))) # sort by probability
    
def most_near_0(results):
  return list(sorted(results, key = lambda x: x[1])) # sort by probability

def draw_by_results_and_index(results, index):
  label, probability, memory_info_copy, sentence_scores, word_scores_per_sentence = results[index]
  if len(word_scores_per_sentence) == 2:
    sentence_att = sentence_scores # (batch, head, n, n)
    sentence_att = sentence_att[0] # (head, n, n)
    sentence_att = sentence_att.mean(0) # (n, n)
    sentence_att = sentence_att[-1] # (n)
    U.draw_head_attention(word_scores_per_sentence[0], memory_info_copy[0], path = 'left.png', desc= f'target: {label}, probability: {probability},  att_score: {sentence_att[0]}')
    U.draw_head_attention(word_scores_per_sentence[1], memory_info_copy[1], path = 'right.png', desc = f'target: {label}, probability: {probability},  att_score: {sentence_att[1]}')
  elif len(word_scores_per_sentence) == 1:
    print(f'Warning: just 1 sentence: {memory_info_copy}')
    U.draw_head_attention(word_scores_per_sentence[0], memory_info_copy[0], path = 'begin.png', desc = 'att_score: 1')
    
