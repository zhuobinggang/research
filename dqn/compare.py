from bert_mlp import BERT_MLP, train as train_mlp, test as test_mlp
from bert_lstm import BERT_LSTM, train as train_lstm, test as test_lstm
from bert_crf import BERT_LSTM_CRF, get_ds, cal_prec_rec_f1_v2, train as train_crf, test as test_crf

mlps = []
lstms = []
crfs = []

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
        m = BERT_CRF()
        _ = train_crf(ds_train, m)
        results, targets = test_crf(ds_test, m)
        crfs.append(cal_prec_rec_f1_v2(results, targets))
    
