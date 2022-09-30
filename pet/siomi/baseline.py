import torch
t = torch
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import datetime
from itertools import chain
from reader import read_data
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score, precision_score, recall_score
import types
from reader import read_data
# from main import create_inputs_and_labels_from_fewshot_set

def create_model(learning_rate = 1e-5):
    res = types.SimpleNamespace()
    res.bert = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    res.opter = torch.optim.AdamW(res.bert.parameters(), learning_rate)
    res.bert.cuda()
    res.bert.train()
    return res

# ======================

def create_inputs_and_labels_from_fewshot_set(ds, shuffle = True):
    results = []
    ds = ds.copy()
    if shuffle:
        np.random.shuffle(ds)
    half = int(len(ds) / 2)
    ds_true = ds[:half]
    ds_false = ds[half:]
    for left, right in ds_true:
        results.append((f'[CLS]{left}[SEP]{right}[SEP]', 0))
    for left, right in ds_false:
        results.append((f'[CLS]{right}[SEP]{left}[SEP]', 1))
    if shuffle:
        np.random.shuffle(results)
    return results

def step(x, y, m):
    toker = m.toker
    bert = m.bert
    opter = m.opter
    # print(f'{x} ==> {y}')
    inputs = toker.encode(x, add_special_tokens=False)
    inputs = torch.LongTensor([inputs]).cuda()
    # logits = bert(inputs).logits # (1, 2)
    labels = torch.LongTensor([y]).cuda() # (1)
    loss = bert(inputs, labels = labels).loss
    loss.backward()

def get_test_result(m, ds_test):
    toker = m.toker
    bert = m.bert
    pred_ys = []
    true_ys = []
    length = len(ds_test)
    input_labels = create_inputs_and_labels_from_fewshot_set(ds_test, shuffle = False)
    # for index, item in enumerate(test):
    for index, (text, label) in enumerate(input_labels):
        if index % 100 == 0:
            print(f'{index}/{length}')
        inputs = toker.encode(text, add_special_tokens=False)
        inputs = torch.LongTensor([inputs]).cuda()
        with torch.no_grad():
            logits = bert(inputs).logits
        pred_ys.append(logits.argmax().item())
        true_ys.append(label)
    return pred_ys, true_ys

def get_test_result_random_baseline(ds_test):
    pred_ys = []
    true_ys = []
    input_labels = create_inputs_and_labels_from_fewshot_set(ds_test, shuffle = False)
    true_ys = [y for x,y in input_labels]
    pred_ys = np.random.randint(0,2, len(true_ys))
    return pred_ys, true_ys

def run_full(m, ds, batch_size = 4):
    input_labels = create_inputs_and_labels_from_fewshot_set(ds)
    length = len(input_labels)
    first_time = datetime.datetime.now()
    for index, (x,y) in enumerate(input_labels):
        step(x, y, m)
        if (index + 1) % 32 == 0:
            print(f'{index + 1} / {length}')
        if (index + 1) % batch_size == 0:
            m.opter.step()
            m.opter.zero_grad()
    m.opter.step()
    m.opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)

def get_curve():
    m = create_model()
    ds = read_data()
    train_ds = ds[:496]
    test_ds = ds[500:700]
    fs = []
    precs = []
    recs = []
    for start in range(0, 496, 16): # [0, 16, 32, ..., 480]
        end = start + 16 # 16, 32, ..., 496
        fewshot = train_ds[start: end]
        run_full(m, fewshot, batch_size = 4)
        results, targets = get_test_result(m, test_ds)
        fs.append(f1_score(targets, results, average='macro'))
        precs.append(precision_score(targets, results, average='macro'))
        recs.append(recall_score(targets, results, average='macro'))
    return precs, recs, fs

def get_curve_random():
    ds = read_data()
    train_ds = ds[:496]
    test_ds = ds[500:700]
    fs = []
    precs = []
    recs = []
    for start in range(0, 496, 16): # [0, 16, 32, ..., 480]
        end = start + 16 # 16, 32, ..., 496
        results, targets = get_test_result_random_baseline(test_ds)
        fs.append(f1_score(targets, results, average='macro'))
        precs.append(precision_score(targets, results, average='macro'))
        recs.append(recall_score(targets, results, average='macro'))
    return precs, recs, fs

def train_model(data_points = 256, batch_size = 4):
    m = create_model()
    ds = read_data()
    assert data_points < 500
    train_ds = ds[:data_points]
    test_ds = ds[500:700]
    print('Training...')
    run_full(m, train_ds, batch_size = batch_size)
    print('Testing...')
    results, targets = get_test_result(m, test_ds)
    f = f1_score(targets, results, average='macro')
    prec = precision_score(targets, results, average='macro')
    rec = recall_score(targets, results, average='macro')
    print(f'Results: f {f}, precision {prec}, recall {rec}.')
    return m


