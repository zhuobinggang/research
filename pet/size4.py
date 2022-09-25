import torch
t = torch
import numpy as np
nn = t.nn
F = t.nn.functional
from reader import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score
from main import Sector_Pet, deverbalize, verbalize
import datetime

ds_train = read_ld_train_from_chapters(4)
ds_test = read_ld_test_from_chapters(4)

def ss_to_patterned_ids(toker, ss):
    pattern_ids = []
    ss = [item for item in ss if item is not None]
    for s in ss:
        pattern_new = f'（新しい段落なのか？[MASK]。）{s}'
        ids = toker.encode(pattern_new, max_length = 120, truncation = True , add_special_tokens = False)
        pattern_ids += ids
        if len(ids) >= 119:
            # print(pattern_new)
            pass
    pattern_ids.insert(0, toker.cls_token_id)
    pattern_ids.append(toker.sep_token_id)
    return pattern_ids

# DONE
# return: ids
def create_inputs_and_labels_from_fewshot_set(fewshot_set, toker):
    patterns = []
    verbalized_patterns = []
    patterns_ids = []
    verbalized_patterns_ids = []
    for case in fewshot_set:
        ss, labels = case
        assert len(ss) == 4
        ss = [item for item in ss if item is not None]
        labels = [item for item in labels if item is not None]
        pattern_ids = []
        pattern_verbalized_ids = []
        for s,label in zip(ss, labels):
            pattern_new = f'（新しい段落なのか？[MASK]。）{s}'
            pattern_verbalized_new = pattern_new.replace('[MASK]', verbalize(label))
            ids = toker.encode(pattern_new, max_length = 120, truncation = True , add_special_tokens = False)
            pattern_ids += ids
            if len(ids) >= 119:
                print(pattern_new)
            pattern_verbalized_ids += toker.encode(pattern_verbalized_new, max_length = 120, truncation = True , add_special_tokens = False)
        pattern_ids.insert(0, toker.cls_token_id)
        pattern_ids.append(toker.sep_token_id)
        patterns_ids.append(pattern_ids)
        pattern_verbalized_ids.insert(0, toker.cls_token_id)
        pattern_verbalized_ids.append(toker.sep_token_id)
        verbalized_patterns_ids.append(pattern_verbalized_ids)
    # return patterns, verbalized_patterns
    return patterns_ids, verbalized_patterns_ids

def step(x, y, m):
    tokenizer = m.toker
    model = m
    opter = m.opter
    inputs = torch.LongTensor([x]).cuda()
    labels = torch.LongTensor([y]).cuda()
    labels = torch.where(inputs == tokenizer.mask_token_id, labels, -100)
    outputs = model.bert(inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

def get_predicted_word(tokenizer, model, ss):
    ids = ss_to_patterned_ids(tokenizer, ss)
    inputs = torch.LongTensor([ids]).cuda()
    with torch.no_grad():
        logits = model(inputs).logits
    mask_token_index = (inputs == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    return word

def run_full(m, train_ds, batch_size = 16):
    patterns_ids, verbalized_patterns_ids = create_inputs_and_labels_from_fewshot_set(train_ds, m.toker)
    length = len(train_ds)
    first_time = datetime.datetime.now()
    for index, (x,y) in enumerate(zip(patterns_ids, verbalized_patterns_ids)):
        step(x, y, m)
        if (index + 1) % 500 == 0:
            print(f'{index + 1} / {length}')
        if (index + 1) % batch_size == 0:
            m.opter.step()
            m.opter.zero_grad()
    m.opter.step()
    m.opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)

def get_test_result(m, ds_test):
    tokenizer = m.toker
    model = m.bert
    pred_ys = []
    true_ys = []
    length = len(ds_test)
    # for index, item in enumerate(test):
    for index, case in enumerate(ds_test):
        ss, labels = case 
        assert len(labels) == 4
        if index % 100 == 0:
            print(f'{index}/{length}')
        words = get_predicted_word(tokenizer, model, ss)
        words = words.split()
        assert len(words) >= 3
        pred_y_redundant = [deverbalize(word) for word in words]
        if labels[0] is None: # 例外
            pred_y = pred_y_redundant[1]
        else:
            pred_y = pred_y_redundant[2]
        pred_ys.append(pred_y)
        true_ys.append(labels[2])
    return pred_ys, true_ys

