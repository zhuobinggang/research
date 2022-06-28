import torch as t
from datasets import load_dataset
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
    self.toker = BertTokenizer.from_pretrained('bert-base-uncased')
    self.lstm = nn.LSTM(768, 256, batch_first = True, bidirectional=True) # TODO: 尝试512和768的隐藏值
    self.mlp = nn.Sequential(
            nn.Linear(512, 768),
            nn.Tanh(),
            nn.Linear(768, 9),
    )
    self.CEL = nn.CrossEntropyLoss()


def train(ds_train, m):
    toker = m.toker
    bert = m.bert
    for row in ds_train:
        tokens_org = row['tokens']
        # DESC: 每个token可能会被分解成多个subword，所以用headword_indexs来获取开头的subword对应的embedding
        tokens, ids, headword_indexs = subword_tokenize(tokens_org)
        out = bert(ids).last_hidden_state 
        out_bert = out[:, headword_indexs, :] # (1, n, 768)
        out_lstm, _ = m.lstm(out_bert) # (1, n, 256 * 2)
        out_mlp = m.mlp(out_lstm) # (1, n, 9)
        ys = F.softmax(out_mlp, dim = 2)
        # cal loss
        labels = F.one_hot(t.tensor(row['ner_tags']), 9).unsqueeze(0).float() # (1, n, 9)
        loss = self.CEL(ys, labels)

        
        


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
    ids = t.tensor(ids).unsqueeze(0)
    return tokens, ids, headword_indexs

