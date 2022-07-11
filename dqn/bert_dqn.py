import torch as t
from datasets import load_dataset
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import datetime
import random

class BERT_DQN(nn.Module):
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
    self.opter = t.optim.Adam(self.parameters(), lr=3e-5)
  def dry_run(self, ids, headword_indexs):
    token_embs = self.bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
    actions = []
    seq_lenth = token_embs.shape[1]
    for i in range(seq_lenth):
        obs = token_embs[0,i] # (768)
        action = self.q_net(obs).argmax().item()
        actions.append(action)
    return actions

# token_embs: (1, n, 768)
# labels: (n)
def step_through_episode(m, token_embs, labels, epsilon = 0.2, batch_size = 1):
    reward_sum = 0
    seq_lenth = token_embs.shape[1]
    for i in range(seq_lenth):
        done = (i == seq_lenth - 1)
        obs = token_embs[0,i] # (768)
        action = np.random.randint(0, 9) if random.random() <= epsilon else m.q_net(obs).argmax().item()
        q_pred = m.q_net(obs)[action]
        reward = 1 if labels[i] == action else -1
        reward_sum += reward
        if done:
            with t.no_grad():
                q_true = m.GAMMA * m.q_net(token_embs[0, i + 1]).max() + reward
        else: 
                q_true = reward
        loss = nn.functional.smooth_l1_loss(q_pred, q_true)
        # step back
        loss.backward()
        if (row_idx + 1) % batch_size == 0:
            opter.step()
            opter.zero_grad()
    # The last step
    opter.step()
    opter.zero_grad()
    return reward_sum

def cal_epsilon(start, end, step, total_steps, end_fraction):
    progress = step / total_steps
    if progress > end_fraction:
        return end
    else:
        return start + progress * (end - start) / end_fraction

def episode(m, row, epsilon, batch_size = 4):
    tokens_org = row['tokens']
    tokens, ids, headword_indexs = subword_tokenize(tokens_org, m.toker)
    out_bert = m.bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
    labels = row['ner_tags']
    reward_sum = step_through_episode(m, out_bert, labels, epsilon = epsilon, batch_size = batch_size)
    return reward_sum

reward_per_episode = []

def train_dqn(ds_train, m, epoch = 1, batch = 4):
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = m.opter
    reward_per_episode.clear()
    # CEL = nn.CrossEntropyLoss(weight=t.tensor([weight, 1, 1, 1, 1, 1, 1, 1, 1.0]).cuda(), reduction='sum')
    total_length = len(ds_train)
    for epoch_idx in range(epoch):
        print(f'MLP epoch {epoch_idx}')
        for row_idx, row in enumerate(np.random.permutation(ds_train)):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            epsilon = cal_epsilon(1, 0.05, row_idx + 1, total_length + 1, 0.5)
            reward_per_episode.append(episode(m, row, epsilon, batch_size = 4))
    last_time = datetime.datetime.now()
    delta = last_time - first_time 
    print(delta.seconds)
    return delta.seconds

