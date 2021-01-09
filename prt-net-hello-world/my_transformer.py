import torch as t
import torch.nn as nn

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size = 64):
    super().__init__()
    self.Query = nn.Linear(input_size, hidden_size)
    self.Key = nn.Linear(input_size, hidden_size)
    self.Value = nn.Linear(input_size, hidden_size)
    self.softmax = t.nn.Softmax(1) # for (seq_len, input_size)

  # embs: (seq_len, input_size)
  def forward(self, embs):
    querys = self.Query(embs) # (seq_len, hidden_size)
    keys = self.Key(embs) # (seq_len, hidden_size)
    values = self.Value(embs) # (seq_len, hidden_size)
    # TODO: calculate score
    scores = t.mm(querys, keys.T) # (seq_len, seq_len)
    focus_degree = self.softmax(scores) # (seq_len, seq_len)
    results_per_query = []
    for degree_per_query in focus_degree: # (seq_len)
      weighted_values_for_query = (degree_per_query * values.T).T.sum(0) # (hidden_size)
      results_per_query.append(weighted_values_for_query)
    results = t.stack(results_per_query)
    return results

    
