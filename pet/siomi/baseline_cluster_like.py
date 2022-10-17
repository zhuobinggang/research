import torch
t = torch
import numpy as np
nn = t.nn
F = t.nn.functional
from baseline import create_model
from reader import customized_ds
import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
import types

train_ds, test_ds = customized_ds()

def processed_input(word):
    return f'[CLS]{word}[SEP]'

def run_full(m, ds, batch_size = 4):
    input_labels = [(processed_input(word), label) for word,label in ds]
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

def step(x, y, m):
    toker = m.toker
    bert = m.bert
    opter = m.opter
    inputs = toker.encode(x, add_special_tokens=False)
    inputs = torch.LongTensor([inputs]).cuda()
    labels = torch.LongTensor([y]).cuda() # (1)
    loss = bert(inputs, labels = labels).loss
    loss.backward()

def get_curve(m, batch_size = 1):
    train_ds, test_ds = customized_ds()
    fs = []
    precs = []
    recs = []
    for start in range(0, len(train_ds), 32): 
        end = start + 32 
        fewshot = train_ds[start: end]
        run_full(m, fewshot, batch_size = batch_size)
        results, targets = get_test_result(m, test_ds)
        fs.append(f1_score(targets, results, average='macro'))
        precs.append(precision_score(targets, results, average='macro'))
        recs.append(recall_score(targets, results, average='macro'))
    return precs, recs, fs

def batch_get_curve(epoch = 9, batch = 1):
    m = create_model()
    fss = []
    for i in range(epoch):
        _,_,fs = get_curve(m, batch)
        fss += fs
    return fss

def get_test_result(m, ds_test):
    toker = m.toker
    bert = m.bert
    pred_ys = []
    true_ys = []
    length = len(ds_test)
    input_labels = [(processed_input(word), label) for word,label in ds_test]
    # for index, item in enumerate(test):
    for index, (text, label) in enumerate(input_labels):
        inputs = toker.encode(text, add_special_tokens=False)
        inputs = torch.LongTensor([inputs]).cuda()
        with torch.no_grad():
            logits = bert(inputs).logits
        pred_ys.append(logits.argmax().item())
        true_ys.append(label)
    return pred_ys, true_ys

def train_model(data_points = 256, batch_size = 4):
    m = create_model()
    assert data_points < 500
    print('Training...')
    run_full(m, train_ds, batch_size = batch_size)
    print('Testing...')
    results, targets = get_test_result(m, test_ds)
    f = f1_score(targets, results, average='macro')
    prec = precision_score(targets, results, average='macro')
    rec = recall_score(targets, results, average='macro')
    print(f'Results: f {f}, precision {prec}, recall {rec}.')
    return m


