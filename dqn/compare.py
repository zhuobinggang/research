from bert_mlp import BERT_MLP, train_by_batch as train_mlp
from bert_lstm import BERT_LSTM, train as train_lstm
from bert_crf import BERT_LSTM_CRF, BERT_MLP_CRF, cal_prec_rec_f1_v2, train as train_crf
from datasets import load_metric, load_dataset
import torch as t
# metric = load_metric('seqeval')
# metric.compute(predictions = y_pred, references = y_true)


mlps = []
lstms = []
lstm_crfs = []
crfs = []

def get_ds():
    ds = load_dataset("conll2003")
    train = ds['train']
    test = ds['test']
    return train, test

keys = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

def idxs2key(idxs):
    return [keys[idx] for idx in idxs]

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

def test(ds_test, m):
    y_true = []
    y_pred = []
    toker = m.toker
    bert = m.bert
    for row_idx, row in enumerate(ds_test):
        tokens_org = row['tokens']
        # DESC: 每个token可能会被分解成多个subword，所以用headword_indexs来获取开头的subword对应的embedding
        tokens, ids, headword_indexs = subword_tokenize(tokens_org, m.toker)
        if tokens is None:
            print('跳过')
        else:
            ys = m.dry_run(ids, headword_indexs)
            y_pred.append(idxs2key(ys))
            y_true.append(idxs2key(row['ner_tags'])) 
    return y_true, y_pred

def run(times = 3, epoch = 2):
    ds_train, ds_test = get_ds()
    metric = load_metric('seqeval')
    for _ in range(times):
        # MLP
        m = BERT_MLP()
        _ = train_mlp(ds_train, m, epoch = epoch)
        y_true, y_pred = test(ds_test, m)
        mlps.append(metric.compute(predictions = y_pred, references = y_true))
        # LSTM
        m = BERT_LSTM()
        _ = train_lstm(ds_train, m, epoch = epoch)
        y_true, y_pred = test(ds_test, m)
        lstms.append(metric.compute(predictions = y_pred, references = y_true))
        # CRF
        m = BERT_LSTM_CRF()
        _ = train_crf(ds_train, m, epoch = epoch)
        y_true, y_pred = test(ds_test, m)
        lstm_crfs.append(metric.compute(predictions = y_pred, references = y_true))
        # CRF
        m = BERT_MLP_CRF()
        _ = train_crf(ds_train, m, epoch = epoch)
        y_true, y_pred = test(ds_test, m)
        crfs.append(metric.compute(predictions = y_pred, references = y_true))
    
