from wikiatt import *
from self_attention import Multihead_SelfAtt, Multihead_Official, Multihead_Official_Scores

class AttMemNet(WikiAttOfficial):
  def init_selfatt_layers(self):
    print(f'Init_selfatt_layers: with head = {self.head}. sentence_integrator will output scores')
    self.sentence_compressor = Multihead_Official(self.feature, self.head)
    self.sentence_integrator = Multihead_Official_Scores(self.feature, self.head)

  def init_working_memory(self):
    self.working_memory = []
    print(f'init_working_memory: memory_size = {self.memory_size}')

  # inpts: [seq_len, (?, feature)], 不定长复数句子
  def cat2memory(self, inpts):
    self.working_memory += inpts

  def memory_self_attention(self):
    # get cls
    cls = self.cls(self.working_memory)
    # self attend
    return self.integrate_sentences_info_with_scores(cls)

  def integrate_sentences_info_with_scores(self, cls):
    seq_len, feature = cls.shape
    cls = cls + self.get_pos_encoding(cls)# NOTE: pos encoding
    integrated, scores = self.sentence_integrator(cls) # (seq_len, feature), (seq_len, seq_len)
    return integrated, scores

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
      self.working_memory = [m for m in self.working_memory if m is not None]
      # print(len(self.working_memory))
    if self.memory_checking:
      print([round(item, 5) for item in score.tolist()])
    
  # inpts: [seq_len, (?, feature)], 不定长复数句子
  # labels: (label, pos)
  def train(self, inpts, labels):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    # cat with working memory
    seq_len = len(inpts)
    self.cat2memory(inpts)
    out, scores = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    scores = scores[-seq_len:]
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

  @t.no_grad()
  def dry_run(self, inpts, labels=None):
    label, pos = labels # (1), LongTensor
    pos = pos.item()
    if GPU_OK:
      inpts = [item.cuda() for item in inpts]
      label = label.cuda()
    # cat with working memory
    seq_len = len(inpts)
    self.cat2memory(inpts)
    out, scores = self.memory_self_attention() # (seq_len + current_memory_size, feature), (seq_len + current_memory_size, seq_len + current_memory_size)
    out = out[-seq_len:] # 剪掉记忆储存部分
    scores = scores[-seq_len:]
    self.memory_arrange(scores[pos])
    o = out[pos] # (feature)
    o = o.view(1, self.feature)
    o = self.classifier(o) # (1, 2)
    self.print_train_info(o, label, -1)
    return o.argmax(1)


def run():
  init_G(4)
  head = 6
  size = 5
  for i in range(9):
    G['m'] = m = AttMemNet(hidden_size = 256, head=head, memory_size = size)
    get_datas(i, 1, f'WikiAttMem length=2:2 epoch = {i}, head = {head}, size = {size}')
  size = 0
  for i in range(5):
    G['m'] = m = AttMemNet(hidden_size = 256, head=head, memory_size = size)
    get_datas(i + 10, 1, f'WikiAttMem length=2:2 epoch = {i}, head = {head}, size = {size}')
  size = 2
  for i in range(5):
    G['m'] = m = AttMemNet(hidden_size = 256, head=head, memory_size = size)
    get_datas(i + 20, 1, f'WikiAttMem length=2:2 epoch = {i}, head = {head}, size = {size}')

  init_G(2)
  size = 5
  for i in range(9):
    G['m'] = m = AttMemNet(hidden_size = 256, head=head, memory_size = size)
    get_datas(i + 30, 1, f'WikiAttMem length=1:1 epoch = {i}, head = {head}, size = {size}')


