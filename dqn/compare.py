from bert_mlp import BERT_MLP, train as train_mlp, test as test_mlp
from bert_lstm import BERT_LSTM, train as train_lstm, test as test_lstm
from bert_crf import BERT_LSTM_CRF, BERT_MLP_CRF, get_ds, cal_prec_rec_f1_v2, train as train_crf, test as test_crf

mlps = []
lstms = []
lstm_crfs = []
crfs = []

def idxs2key(idxs):
    return [keys[idx] for idx in idxs]

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
            ys = m.forward(ids, headword_indexs)
            y_pred.append(idxs2key(ys))
            y_true.append(idxs2key(row['ner_tags'])) 
    return y_true, y_pred

def run(times = 5):
    ds_train, ds_test = get_ds()
    for _ in range(times):
        # MLP
        m = BERT_MLP()
        _ = train_mlp(ds_train, m)
        results, targets = test_mlp(ds_test, m)
        mlps.append(cal_prec_rec_f1_v2(results, targets))
        # LSTM
        m = BERT_LSTM()
        _ = train_lstm(ds_train, m)
        results, targets = test_lstm(ds_test, m)
        lstms.append(cal_prec_rec_f1_v2(results, targets))
        # CRF
        m = BERT_LSTM_CRF()
        _ = train_crf(ds_train, m)
        results, targets = test_crf(ds_test, m)
        lstm_crfs.append(cal_prec_rec_f1_v2(results, targets))
        # CRF
        m = BERT_MLP_CRF()
        _ = train_crf(ds_train, m)
        results, targets = test_crf(ds_test, m)
        crfs.append(cal_prec_rec_f1_v2(results, targets))
    
