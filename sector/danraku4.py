import danraku3 as model
import torch as t
import word2vec_fucker as data
from transformers import BertModel, BertJapaneseTokenizer
from torch.nn.utils.rnn import pad_sequence
from itertools import chain

sentence_emb_size = 128

# inpts: (batch_size, (left, right))
# return: (batch_size, sentence_emb_size * 2)
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
  return hn.view(len(inpts), sentence_emb_size * 2)

class Model_Wiki2vec(model.BERT_Cat_Sentence):
  def get_outs(self, inpts):
    embs = get_embs_from_inpts(self, inpts) # (batch_size, 768 * 2)
    outs = self.fw(embs) # (batch_size, 768)
    outs = self.minify(outs) # (batch_size, 1)
    return outs

  def get_should_update(self):
    return chain(self.fw.parameters(), self.minify.parameters(), self.sentence_pooler.parameters())

  def init_hook(self):
    self.fw = t.nn.Linear(sentence_emb_size * 2, sentence_emb_size)
    self.minify = t.nn.Linear(sentence_emb_size, 1)
    self.sentence_pooler = t.nn.LSTM(300, sentence_emb_size, batch_first=True)
