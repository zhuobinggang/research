import torch as t
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size = 64):
    super().__init__()
    self.Query = nn.Linear(input_size, hidden_size)
    self.Key = nn.Linear(input_size, hidden_size)
    self.Value = nn.Linear(input_size, hidden_size)
    self.softmax = t.nn.Softmax(1) # for (seq_len, input_size)

  # (m, n) = (seq_len, hidden_size)
  def forward(self, embs):
    querys = self.Query(embs) # (m, n)
    keys = self.Key(embs) # (m, n)
    values = self.Value(embs) # (m, n)
    # calculate score
    for_softmax = t.mm(querys, keys.T) # (m, m)
    scores = self.softmax(for_softmax) # (m, m)
    results_per_query = []
    for scores_per_query in scores: # (m)
      weighted_values_per_query = (scores_per_query * values.T).T.sum(0) # (n)
      results_per_query.append(weighted_values_per_query)
    results = t.stack(results_per_query) # (m, n)
    return results

    
