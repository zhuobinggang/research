import torch as t
import numpy as np
nn = t.nn

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
