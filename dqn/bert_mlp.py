import torch as t
import numpy as np
nn = t.nn
F = t.nn.functional
# 放弃使用word2vec了，太多单词不存在
# from wikipedia2vec import Wikipedia2Vec
# wiki2vec = Wikipedia2Vec.load('/usr01/ZhuoBinggang/enwiki_20180420_win10_300d.pkl')
from transformers import BertTokenizer, BertModel, BertTokenizerFast
import datetime


class BERT_MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.bert.train()
    self.toker = BertTokenizer.from_pretrained('bert-base-uncased')
    self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 9),
    )
    self.cuda()

  def dry_run(self, ids, headword_indexs):
    out_bert = self.bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
    out_mlp = self.mlp(out_bert) # (1, n, 9)
    ys = F.softmax(out_mlp, dim = 2) # (1, n, 9)
    ys = ys.argmax(2).squeeze(0).tolist()
    return ys

def pad_idss(idss):
    PAD_TOKEN_ID = 0
    max_length = np.max([len(ids) for ids in idss])
    idss_padded = []
    masks = []
    if hasattr(idss[0], 'tolist'):
        idss = [ids.tolist() for ids in idss]
    for ids in idss:
        idss_padded.append(ids + [0] * (max_length - len(ids)))
        masks.append([1] * len(ids) + [0] * (max_length - len(ids)))
    return idss_padded, masks


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batched_random_dataset(dataset, batch_size = 4):
    return batch(np.random.permutation(dataset), batch_size)

def train_by_batch(ds_train, m, epoch = 1, batch = 4, weight = True):
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = t.optim.Adam(m.parameters(), lr=2e-5)
    if weight:
        CEL = nn.CrossEntropyLoss(weight=t.tensor([0.1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda())
    else:
        CEL = nn.CrossEntropyLoss()
    for epoch_idx in range(epoch):
        print(f'MLP epoch {epoch_idx}')
        for row_idx, row in enumerate(np.random.permutation(ds_train)):
            if row_idx % 1000 == 0:
                # print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            tokens_org = row['tokens']
            # DESC: 每个token可能会被分解成多个subword，所以用headword_indexs来获取开头的subword对应的embedding
            tokens, ids, headword_indexs = subword_tokenize(tokens_org, m.toker)
            if tokens is None:
                print('跳过训练')
            else:
                out_bert = bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
                out_mlp = m.mlp(out_bert) # (1, n, 9)
                ys = F.softmax(out_mlp, dim = 2) # (1, n, 9)
                # cal loss
                labels = t.LongTensor(row['ner_tags']) # Long: (n)
                loss = CEL(ys.squeeze(0), labels.cuda())
                loss.backward() # 累积模拟batch
                # backward
                if (row_idx + 1) % batch == 0:
                    opter.step()
                    opter.zero_grad()
    # 最后一次backward
    opter.step()
    opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time 
    print(delta.seconds)
    return delta.seconds

def train(ds_train, m, epoch = 1):
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = t.optim.Adam(m.parameters(), lr=2e-5)
    CEL = nn.CrossEntropyLoss(weight=t.tensor([0.1, 1, 1, 1, 1, 1, 1, 1, 1]).cuda())
    # CEL = nn.CrossEntropyLoss()
    for epoch_idx in range(epoch):
        print(f'MLP epoch {epoch_idx}')
        for row_idx, row in enumerate(np.random.permutation(ds_train)):
            if row_idx % 1000 == 0:
                # print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            tokens_org = row['tokens']
            # DESC: 每个token可能会被分解成多个subword，所以用headword_indexs来获取开头的subword对应的embedding
            tokens, ids, headword_indexs = subword_tokenize(tokens_org, m.toker)
            if tokens is None:
                print('跳过训练')
            else:
                out_bert = bert(ids.cuda()).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
                out_mlp = m.mlp(out_bert) # (1, n, 9)
                ys = F.softmax(out_mlp, dim = 2) # (1, n, 9)
                # cal loss
                labels = t.LongTensor(row['ner_tags']) # Long: (n)
                loss = CEL(ys.squeeze(0), labels.cuda())
                # backward
                m.zero_grad()
                loss.backward()
                opter.step()
    last_time = datetime.datetime.now()
    delta = last_time - first_time 
    print(delta.seconds)
    return delta.seconds
    

# Checked, 可以放心使用, 可以运行test_subword_tokenize尝试
def subword_tokenize(tokens_org, toker):
    tokens_org = [token.lower() for token in tokens_org]
    headword_indexs = []
    tokens = []
    index = 0
    for token in tokens_org:
        sub_tokens = toker.tokenize(token)
        tokens += sub_tokens
        headword_indexs.append(index)
        index += len(sub_tokens)
    if len(tokens) < 1:
        print(f'解码出来的tokens数量为0, {tokens_org}')
        return None, None, None
    else:
        ids = toker.encode(tokens) 
        # NOTE: BUG fixed, encode的时候会增加[cls][sep]，因为cls是增加在左边的，所以headword需要加一
        headword_indexs = [idx + 1 for idx in headword_indexs]
        ids = t.tensor(ids).unsqueeze(0)
        return tokens, ids, headword_indexs

def test_subword_tokenize(tokens_org, toker):
    tokens, ids, headword_indexs = subword_tokenize(tokens_org, toker)
    ids = ids.squeeze()
    id_heads = ids[headword_indexs]
    print(' '.join(tokens_org))
    print(toker.decode(id_heads))



def cal_prec_rec_f1_v2(results, targets):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  for guess, target in zip(results, targets):
    if guess == 1:
      if target == 1:
        TP += 1
      elif target == 0:
        FP += 1
    elif guess == 0:
      if target == 1:
        FN += 1
      elif target == 0:
        TN += 1
  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  rec = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
  balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
  balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
  return prec, rec, f1, balanced_acc
