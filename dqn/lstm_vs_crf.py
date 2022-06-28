import torch as t
from datasets import load_dataset
import numpy as np
nn = t.nn
F = t.nn.functional
# 放弃使用word2vec了，太多单词不存在
# from wikipedia2vec import Wikipedia2Vec
# wiki2vec = Wikipedia2Vec.load('/usr01/ZhuoBinggang/enwiki_20180420_win10_300d.pkl')
from transformers import BertTokenizer, BertModel, BertTokenizerFast


def get_ds():
    ds = load_dataset("conll2003")
    test = ds['test']
    train = ds['test']
    return train, test


class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-uncased')
    self.bert.train()
    self.toker = BertTokenizer.from_pretrained('bert-base-uncased')
    self.lstm = nn.LSTM(768, 256, batch_first = True, bidirectional=True) # TODO: 尝试512和768的隐藏值
    self.mlp = nn.Sequential(
            nn.Linear(512, 768),
            nn.Tanh(),
            nn.Linear(768, 9),
    )
    self.CEL = nn.CrossEntropyLoss()


def train(ds_train, m, epoch = 1):
    toker = m.toker
    bert = m.bert
    opter = t.optim.Adam(m, lr=2e-5)
    for _ in epoch:
        for row in np.random.permutation(ds_train):
            tokens_org = row['tokens']
            # DESC: 每个token可能会被分解成多个subword，所以用headword_indexs来获取开头的subword对应的embedding
            tokens, ids, headword_indexs = subword_tokenize(tokens_org)
            out_bert = bert(ids).last_hidden_state[:, headword_indexs, :] # (1, n, 768)
            out_lstm, _ = m.lstm(out_bert) # (1, n, 256 * 2)
            out_mlp = m.mlp(out_lstm) # (1, n, 9)
            ys = F.softmax(out_mlp, dim = 2) # (1, n, 9)
            # cal loss
            labels = t.LongTensor(row['ner_tags']) # Long: (n)
            loss = self.CEL(ys.squeeze(0), labels)
            # backward
            m.zero_grad()
            loss.backward()
            opter.step()
    


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
