import torch as t
import numpy as np
nn = t.nn
import logging

class Self_Att(nn.Module):
  def __init__(self, feature):
    super().__init__()
    self.feature = feature
    self.squre_dk = np.sqrt(self.feature)
    self.WQ = nn.Linear(feature, feature)
    self.WK = nn.Linear(feature, feature)
    self.WV = nn.Linear(feature, feature)

  def with_score(self, embs):
    qs = self.WQ(embs) # (seq_len, feature)
    ks = self.WK(embs) # (seq_len, feature)
    vs = self.WV(embs) # (seq_len, feature)
    scores = t.mm(qs, ks.transpose(0, 1)) # (seq_len, seq_len)
    scores = t.softmax(scores / self.squre_dk, 1) # (seq_len, seq_len)
    result = t.mm(scores, vs) # (seq_len, feature)
    return result, scores.detach()

  # embs: (seq_len, feature)
  # return: (seq_len, feature)
  def forward(self, embs):
    res, scores = self.with_score(embs)
    return res,scores


class Self_Att2(Self_Att):
  # embs: (seq_len, feature)
  # return: (seq_len, feature)
  def forward(self, embs):
    res, scores = self.with_score(embs)
    return res

class Multihead_SelfAtt(nn.Module):
  def __init__(self, feature, head):
    super().__init__()
    if not (feature / head).is_integer():
      logging.error("NONONONONO! (feature / head) is not integer")
      return
    self.head = head
    self.feature = feature
    self.dk = int(feature / head)
    self.squre_dk = np.sqrt(self.dk)
    self.WQ = nn.Linear(feature, feature)
    self.WK = nn.Linear(feature, feature)
    self.WV = nn.Linear(feature, feature)
    self.WO = nn.Linear(feature, feature)

  # return:
  # results: (seq_len, feature)
  # scores: (heads, seq_len, seq_len)
  def with_score(self, embs):
    qs = self.WQ(embs) # (seq_len, feature)
    ks = self.WK(embs) # (seq_len, feature)
    vs = self.WV(embs) # (seq_len, feature)
    qss = qs.split(self.dk, 1) # [head, (seq_len, feature / head)]
    kss = ks.split(self.dk, 1) # [head, (seq_len, feature / head)]
    vss = vs.split(self.dk, 1) # [head, (seq_len, feature / head)]
    scores = []
    results = []
    for qi,ki,vi in zip(qss, kss, vss):
      score = t.mm(qi, ki.transpose(0, 1)) # (seq_len, seq_len)
      score = t.softmax(score / self.squre_dk, 1) # (seq_len, seq_len)
      result = t.mm(score, vi) # (seq_len, feature / head)
      scores.append(score.detach())
      results.append(result)
    results = t.cat(results, 1) # (seq_len, feature)
    results = self.WO(results)
    scores = t.stack(scores)
    return results, scores

  # return: (seq_len, feature)
  def forward(self, embs):
    res, scores = self.with_score(embs)
    return res


class Multihead_Official(nn.Module):
  def __init__(self, feature, head):
    super().__init__()
    self.feature = feature
    self.head = head
    self.main = nn.MultiheadAttention(feature, head)

  # return:
  # results: (seq_len, feature)
  # scores: (heads, seq_len, seq_len)
  def with_score(self, embs):
    embs = embs.view(-1, 1, self.feature)
    out, scores = self.main(embs)
    return out.view(-1, self.feature), scores

  # return: (seq_len, feature)
  def forward(self, embs):
    embs = embs.view(-1, 1, self.feature)
    out, scores = self.main(embs)
    return out.view(-1, self.feature)

