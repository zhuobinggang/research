from main import create_model, get_predicted_word
from reader import read_data, read_regular_ds, customized_ds
import numpy as np
import datetime
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

PROMPT = 0

def verbalize(label):
    if PROMPT == 0:
        return '柔らかい' if label == 0 else '硬い'
    elif PROMPT == 1:
        return 'いえ' if label == 0 else 'はい'

def deverbalize(word):
    if PROMPT == 0:
        if word == '柔らかい':
            return 0
        elif word == '硬い':
            return 1
        else:
            print(f'Bad word: {word}')
            return 0
    elif PROMPT == 1:
        if word == 'いえ':
            return 0
        elif word == 'はい':
            return 1
        else:
            print(f'Bad word: {word}')
            return 0

def step(x, y, m):
    tokenizer = m.toker
    model = m
    opter = m.opter
    inputs = tokenizer(x, return_tensors="pt", truncation=True)["input_ids"]
    labels = tokenizer(y, return_tensors="pt", truncation=True)["input_ids"]
    labels = torch.where(inputs == tokenizer.mask_token_id, labels, -100)
    inputs = inputs.cuda()
    labels = labels.cuda()
    outputs = model.bert(inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

def pattern(word):
    if PROMPT == 0:
        return f'「{word}」とは[MASK]表現です。'
    elif PROMPT == 1:
        return f'「{word}」とは堅い表現ですか？[MASK]。'

def ds_texted(ds_org, deverbalize_label = False):
    ds = []
    for word, label in ds_org:
        # x_text = f'「{word}」は上品な言葉ですか？[MASK]。'
        # x_text = f'「{word}」とはよく使われている言葉ですか？[MASK]。'
        x_text = pattern(word)
        y = x_text.replace('[MASK]', verbalize(label)) if deverbalize_label else label
        ds.append((x_text, y))
    return ds

def run_full(m, ds_org, batch_size = 4):
    ds = ds_texted(ds_org, deverbalize_label = True)
    length = len(ds)
    first_time = datetime.datetime.now()
    for index, (x,y) in enumerate(ds):
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

def get_test_result(m, ds_org):
    toker = m.toker
    bert = m.bert
    pred_ys = []
    true_ys = []
    length = len(ds_org)
    ds = ds_texted(ds_org, deverbalize_label = False)
    # for index, item in enumerate(test):
    for index, (x_text, label) in enumerate(ds):
        if index % 100 == 0:
            print(f'{index}/{length}')
        word = get_predicted_word(toker, bert, x_text)
        pred_y = deverbalize(word)
        pred_ys.append(pred_y)
        true_ys.append(label)
    return pred_ys, true_ys

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

def batch_get_curve(epoch = 9):
    m = create_model()
    fss = []
    for i in range(epoch):
        _,_,fs = get_curve(m, 1)
        fss += fs
    return fss


def stop_when_f_score_exceed(m, limit = 0.6, batch_size = 1):
    train_ds, test_ds = customized_ds()
    fs = []
    precs = []
    recs = []
    for start in range(0, len(train_ds), 32): 
        end = start + 32 
        fewshot = train_ds[start: end]
        run_full(m, fewshot, batch_size = batch_size)
        results, targets = get_test_result(m, test_ds)
        f = f1_score(targets, results, average='macro')
        prec = precision_score(targets, results, average='macro')
        rec = recall_score(targets, results, average='macro')
        fs.append(f)
        precs.append(prec)
        recs.append(rec)
        if f > limit:
            print(f'End at {start}, f {f}, prec {prec}, rec {rec}.')
            break
    return precs, recs, fs


def train_model(data_points = 500, batch_size = 4):
    m = create_model()
    train_ds, test_ds = customized_ds()
    train_ds = train_ds[:data_points]
    print('Training...')
    run_full(m, train_ds, batch_size = batch_size)
    print('Testing...')
    results, targets = get_test_result(m, test_ds)
    f = f1_score(targets, results, average='macro')
    prec = precision_score(targets, results, average='macro')
    rec = recall_score(targets, results, average='macro')
    print(f'Results: f {f}, precision {prec}, recall {rec}.')
    return m


