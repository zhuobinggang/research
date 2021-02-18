import danraku4 as parent
import torch as t
from itertools import chain
import word2vec_fucker as data
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

sentence_emb_size = 128

class FakeSector(parent.Model_Wiki2vec):
  # inpts: (batch_size, (left, right))
  # return: (batch_size * 2, sentence_emb_size)
  def get_embs_from_inpts(self, inpts):
    batches = [] 
    for left_ss, right_ss in inpts:
      left_ss = '。'.join(left_ss)
      right_ss = '。'.join(right_ss)
      left_vecs = [t.from_numpy(vec) for vec in data.sentence_to_wordvecs(left_ss)]
      left_wordvecs = t.stack(left_vecs) if len(left_vecs) > 0 else t.zeros(1, 300) # (?, 300)
      right_vecs = [t.from_numpy(vec) for vec in data.sentence_to_wordvecs(right_ss)]
      right_wordvecs = t.stack(right_vecs) if len(right_vecs) > 0 else t.zeros(1, 300)  # (?, 300)
      batches.append(left_wordvecs)
      batches.append(right_wordvecs) 
    # batches: (batch_size * 2, ?, 300)
    pad_batches = pad_sequence(batches, batch_first = True).detach() # (batch_size * 2, max_len, 300)
    _, (hn, _) = self.sentence_pooler(pad_batches) # (1, batch_size * 2, sentence_emb_size)
    return hn.view(len(inpts) * 2, sentence_emb_size)

  def get_outs(self, inpts):
    return self.minify(self.fw(self.get_embs_from_inpts(inpts))) # (batch_size * 2, 1)

  def fake_sector_labels(self, ground_truth, labels_processed):
    fake_sector_labels = []
    for idx,label in enumerate(labels_processed):
      value = ground_truth[idx].item() 
      if label == 1: # diff
        if value < 0:
          fake_sector_labels.append(1)
        else:
          fake_sector_labels.append(0)
      elif label == 0: # same
        if value < 0:
          fake_sector_labels.append(0)
        else:
          fake_sector_labels.append(1)
    fake_sector_labels = t.LongTensor(fake_sector_labels)
    return fake_sector_labels

  def get_loss(self, inpts, labels):
    outs = self.get_outs(inpts) # (batch_size * 2, 1)
    ground_truth = outs[0::2] # (batch_size, 1)
    labels_processed = self.labels_processed(labels, None) # (batch_size)
    fake_sector_labels = self.fake_sector_labels(ground_truth, labels_processed)
    to_predict = outs[1::2] # (batch_size, 1)
    loss = self.get_loss_by_input_and_target(to_predict, fake_sector_labels)
    self.print_info_this_step(to_predict, fake_sector_labels, loss)
    return loss

  def get_should_update(self):
    return chain(self.fw.parameters(), self.minify.parameters(), self.sentence_pooler.parameters())

  def init_hook(self):
    self.fw = t.nn.Linear(sentence_emb_size, int(sentence_emb_size / 2))
    self.minify = t.nn.Linear(int(sentence_emb_size / 2), 1)
    self.sentence_pooler = t.nn.LSTM(300, sentence_emb_size, batch_first=True)
    self.weight_one = 3
    self.weight_zero = 1

  @t.no_grad()
  def dry_run(self, inpts):
    out = self.get_outs(inpts) # (2, 1)
    return 1 if out[0].item() * out[1].item() > 0 else 0

class FakeSectorDot(FakeSector):
  def get_should_update(self):
    return chain(self.sentence_pooler.parameters())

  # (batch_size, 1)
  def get_outs(self, inpts):
    embs = self.get_embs_from_inpts(inpts) # (batch_size * 2, sentence_emb_size)
    # 1) dot 2) sigmoid
    scores = []
    for left,right in zip(embs[0::2], embs[1::2]): # (batch_size, sentence_emb_size)
      scores.append(t.dot(left, right)) 
    return t.stack(scores).view(-1, 1) # (batch_size, 1)

  def get_loss(self, inpts, labels):
    outs = self.get_outs(inpts) # (batch_size, 1)
    labels_processed = self.labels_processed(labels, None) # (batch_size)
    loss = self.get_loss_by_input_and_target(outs, labels_processed)
    self.print_info_this_step(outs, labels_processed, loss)
    return loss

  @t.no_grad()
  def dry_run(self, inpts):
    out = self.get_outs(inpts) # (1, 1)
    return 1 if out.item() > 0 else 0


