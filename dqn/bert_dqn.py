import torch as t
from datasets import load_dataset
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import datetime
import random

class BERT_LSTM(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.bert.train()
    self.toker = BertTokenizer.from_pretrained('bert-base-uncased')
    self.q_net = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 9),
    )
    self.GAMMA = 0.99
    self.cuda()

  def forward(self, obs): # return: (9)
      # TODO:
      pass

  def dry_run(self, ids, headword_indexs):
    out_bert = self.bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
    out_lstm, _ = self.lstm(out_bert) # (1, n, 256 * 2)
    out_mlp = self.mlp(out_lstm) # (1, n, 9)
    ys = F.softmax(out_mlp, dim = 2) # (1, n, 9)
    ys = ys.argmax(2).squeeze(0).tolist()
    return ys

# token_embs: (1, n, 768)
# labels: (n)
def logic(m, token_embs, labels, epsilon = 0.2):
    seq_lenth = token_embs.shape[1]
    for i in range(seq_lenth)
        done = (i == seq_lenth - 1)
        obs = token_embs[i]
        action = np.random.randint(0, 9) if random.random() <= epsilon else m(obs).argmax().item()
        q_pred = m(obs)[action]
        reward = 1 if labels[i] == action else -1
        if done:
            with t.no_grad():
                q_true = m.GAMMA * m(token_embs[i + 1]).max() + reward
        else: 
                q_true = reward
        loss = nn.functional.smooth_l1_loss(q_pred, q_true)
        # TODO: step back





