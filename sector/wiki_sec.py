import word2vec_fucker as W
import data_jap_reader as data
import mainichi

G = {}
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)

def init_G_Symmetry_Mainichi_With_Valid(half = 1, batch = 4, mini = False):
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_trains(mini))
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_tests(mini))
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_valid(mini))
  G['validld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

def fit_embs_to_label(embs):
  return embs.argmax(1)

# =====================================

class Model_GRU(nn.Module):
  def __init__(self, hidden_size = 256, weight_one = 1):
    super().__init__()
    self.hidden_size = hidden_size
    self.CEL = nn.CrossEntropyLoss(t.FloatTensor([1, weight_one]))
    self.wordvec_size = 300
    self.optim = optim.AdamW(self.get_should_update(), self.learning_rate())
    print(f'Init AdamW with lr = {self.learning_rate()}')
    if GPU_OK:
      _ = self.cuda()

  def learning_rate(self):
    return 1e-3

  def init_hook(self):
    self.gru1 = t.nn.GRU(self.wordvec_size, self.hidden_size, batch_first=True) # Integrate words
    self.gru2 = t.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, bidirectional=True) # Integrate sentence 
    self.classifier = nn.Sequential(
      nn.Linear(self.hidden_size * 2, self.hidden_size),
      nn.LeakyReLU(0.1),
      nn.Linear(self.hidden_size, 2),
    )

  def get_should_update(self):
    return chain(self.gru1.parameters(), self.gru2.parameters(), self.classifier.parameters())

  def handle_mass(self, mass): # 和之前的不一样，这个idea需要保留所有labels
    sss = []
    labels = []
    pos = []
    for ss,l,p in mass:
      if len(ss) != self.ss_len_limit:
        print(f'skip: {ss}')
      else:
        sss.append(ss)
        labels.append(l[p]) # s1 [sep1] s2 [sep2] s3 [sep3] => [sep1, sep2]
        pos.append(p)
    return sss, labels, pos

  def cal_los(self, embs, labels):
    assert len(embs.shape) == 2 and embs.shape[1] == 2
    labels = t.LongTensor(labels).to(dev)
    assert len(labels.shape) == 1
    return self.CEL(embs, labels)

  def classfy(self, embs):
    embs = t.stack(embs)
    assert len(embs.shape) == 2 and embs.shape[1] == self.hidden_size * 2
    return self.classifier(embs)

  # sentence_embs: (seq_len, hidden_size)
  def integrate_sentences(self, sentence_embs): 
    embs = sentence_embs
    assert len(embs.shape) == 2 and embs.shape[1] == self.hidden_size
    out, _ = self.gru2(embs.view(embs.shape[0], 1, self.hidden_size)) # (seq_len, 1, hidden_size * 2)
    return out.view(embs.shape[0], self.hidden_size * 2)

  def integrate_words(self, s): 
    vecs = W.sentence_to_wordvecs(s, max_len = 64)
    if len(vecs) < 1: 
      print(f'No words valid: {s}')
      return t.zeros(self.wordvec_size)
    else:
      vecs = t.stack(vecs) # (?, 300)
      assert len(vecs.shape) == 2 and vecs.shape[1] == self.wordvec_size
      _, hn = self.gru1(vecs.view(1, vecs.shape[0], self.wordvec_size)) # (1, 1, hidden_size)
      return hn.view(self.wordvec_size)

  def train(self, mess):
    sss, labels, poss = self.handle_mass(mess) 
    if len(sss) < 1:
      print('No train')
    else:
      embs = []
      for ss in sss:
        sentence_embs = []
        for s in ss: 
          sentence_embs.append(self.integrate_words(s))
        embs.append(self.integrate_sentences(sentence_embs))
      embs = self.classfy(embs)
      loss = self.cal_los(embs, labels)
      self.zero_grad()
      loss.backward()
      self.optim.step()
      return loss.detach().item()
      
  def dry_run(self, mess):
    sss, labels, poss = self.handle_mass(mess) 
    if len(sss) < 1:
      return [], []
    else:
      embs = []
      for ss in sss:
        sentence_embs = []
        for s in ss: 
          sentence_embs.append(self.integrate_words(s))
        embs.append(self.integrate_sentences(sentence_embs))
      embs = self.classfy(embs)
      return t.LongTensor(fit_embs_to_label(embs)), t.LongTensor(labels)


