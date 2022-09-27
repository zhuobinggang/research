import torch
t = torch
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import datetime
from itertools import chain
from reader import read_data
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score, precision_score, recall_score

class Pet(nn.Module):
  def __init__(self, learning_rate = 1e-5):
    super().__init__()
    self.learning_rate = learning_rate
    self.bert_size = 768
    self.init_bert()
    self.opter = t.optim.AdamW(self.get_should_update(), self.learning_rate)
    self.cuda()

  def init_bert(self):
    self.bert = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

  def get_should_update(self):
    return self.bert.parameters()

def verbalize(label):
    if label == 0:
        return 'いえ'
    elif label == 1:
        return 'はい'

def deverbalize(word):
    if word == 'いえ':
        return 0
    elif word == 'はい':
        return 1
    else:
        return word


def patterned(left, right):
    return f'「{left}」は「{right}」より上品ですか？[MASK]。'

def pattern_verbalized(left, right, label):
    text = patterned(left, right)
    return text.replace('[MASK]', verbalize(label))

# NOTE: create half fake datas
def create_inputs_and_labels_from_fewshot_set(ds, should_deverbalize = True, shuffle = True):
    results = []
    ds = ds.copy()
    if shuffle:
        np.random.shuffle(ds)
    half = int(len(ds) / 2)
    ds_true = ds[:half]
    ds_false = ds[half:]
    for left, right in ds_true:
        results.append((patterned(left, right), pattern_verbalized(left, right, 1) if should_deverbalize else 1))
    for left, right in ds_false:
        results.append((patterned(right, left), pattern_verbalized(right, left,  0) if should_deverbalize else 0))
    if shuffle:
        np.random.shuffle(results)
    return results

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

def get_predicted_word(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)["input_ids"]
    inputs = inputs.cuda()
    with torch.no_grad():
        logits = model(inputs).logits
    mask_token_index = (inputs == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    return word



def get_test_result(m, ds_test, should_deverbalize = True):
    tokenizer = m.toker
    model = m.bert
    pred_ys = []
    true_ys = []
    length = len(ds_test)
    input_labels = create_inputs_and_labels_from_fewshot_set(ds_test, should_deverbalize = False, shuffle = False)
    # for index, item in enumerate(test):
    for index, (text, label) in enumerate(input_labels):
        if index % 100 == 0:
            print(f'{index}/{length}')
        word = get_predicted_word(tokenizer, model, text)
        pred_y = deverbalize(word) if should_deverbalize else word
        pred_ys.append(pred_y)
        true_ys.append(label)
    return pred_ys, true_ys

def cal_scores(pred_ys, true_ys):
    precision, recall, fscore, support = score(true_ys, pred_ys)

def cal_prec_rec_f1_v2(results, targets):
    return f1_score(targets, results, average='macro')

def get_curve():
    m = Pet()
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
        
# ========================================

def train_model(data_points = 256, batch_size = 4):
    m = Pet()
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
    print('Results: f {f}, precision {prec}, recall {rec}.')
    return m

def get_predicted_word_by_word_pair(m, left, right):
    text = patterned(left, right)
    return get_predicted_word(m.toker, m.bert, text)

