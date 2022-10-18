import torch
t = torch
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertJapaneseTokenizer, BertForMaskedLM
import datetime
from itertools import chain
from reader import read_ld_train_from_chapters, read_ld_test_from_chapters, read_ld_dev_from_chapters
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score, precision_score, recall_score

ds_train = read_ld_train_from_chapters(2)
ds_test = read_ld_test_from_chapters(2)

RANDOM_SEED = 2022

class Sector_Pet(nn.Module):
  def __init__(self, learning_rate = 1e-5):
    super().__init__()
    self.learning_rate = learning_rate
    self.bert_size = 768
    self.verbose = False
    self.init_bert()
    self.init_hook()
    self.opter = t.optim.AdamW(self.get_should_update(), self.learning_rate)
    self.cuda()

  def init_bert(self):
    self.bert = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters())

  def init_hook(self): 
    self.classifier = nn.Sequential( # 因为要同时判断多种1[sep]3, 2[sep]2, 3[sep]1, 所以多加一点复杂度
      nn.Linear(self.bert_size, 20),
      nn.LeakyReLU(0.1),
      nn.Linear(20, 20),
      nn.LeakyReLU(0.1),
      nn.Linear(20, 1),
      nn.Sigmoid()
    )

def verbalize(label):
    if label == 0:
        return 'いえ'
    else:
        return 'はい'

def deverbalize(word):
    word = word.replace(' ','')
    if word == 'いえ':
        return 0
    elif word == 'はい':
        return 1
    else:
        print(f'Turned {word} to 0')
        return 0

def patterned(ss):
    return f'{ss[0]}（新しい段落ですか？[MASK]。）{ss[1]}'

def case_to_xy(case):
    ss, labels = case
    assert len(ss) == 2
    pattern = f'{ss[0]}（新しい段落ですか？[MASK]。）{ss[1]}'
    pattern_verbalized = pattern.replace('[MASK]', verbalize(labels[1]))
    return pattern, pattern_verbalized

def random_choice_fewshot_training_set(ds, size = 32):
    low = 0
    high = len(ds) - 1
    array = np.random.randint(low, high, size = size)
    results = []
    for index in array:
        results.append(ds[int(index)])
    return results

def sample_until_label_equals(ds, label, pos = 1):
    low = 0
    high = len(ds)
    case = ds[np.random.randint(low, high)]
    while case[1][pos] != label:
        case = ds[np.random.randint(low, high)]
    return case

def random_choice_fewshot_training_set_balanced(ds, size = 32):
    results = []
    for i in range(int(size / 2)):
        results.append(sample_until_label_equals(ds, 0))
        results.append(sample_until_label_equals(ds, 1))
    np.random.shuffle(results)
    return results 

def create_inputs_and_labels_from_fewshot_set(fewshot_set):
    patterns = []
    verbalized_patterns = []
    for item in fewshot_set:
        pattern, verbalized_pattern = case_to_xy(item)
        patterns.append(pattern)
        verbalized_patterns.append(verbalized_pattern)
    return patterns, verbalized_patterns

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

def get_predicted_word(tokenizer, model, ss):
    inputs = tokenizer(patterned(ss), return_tensors="pt", truncation=True)["input_ids"]
    inputs = inputs.cuda()
    with torch.no_grad():
        logits = model(inputs).logits
    mask_token_index = (inputs == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    word = tokenizer.decode(predicted_token_id)
    return word

def run(m, train_ds, size = 1000, batch_size = 16):
    fewshot = random_choice_fewshot_training_set_balanced(train_ds, size = size)
    patterns, verbalized_patterns = create_inputs_and_labels_from_fewshot_set(fewshot)
    for index, (x,y) in enumerate(zip(patterns, verbalized_patterns)):
        step(x, y, m)
        if (index + 1) % 100 == 0:
            print(f'{index + 1} / {size}') 
        if (index + 1) % batch_size == 0:
            m.opter.step()
            m.opter.zero_grad()
    m.opter.step()
    m.opter.zero_grad()

def run_full(m, train_ds, batch_size = 16):
    patterns, verbalized_patterns = create_inputs_and_labels_from_fewshot_set(train_ds)
    length = len(train_ds)
    first_time = datetime.datetime.now()
    for index, (x,y) in enumerate(zip(patterns, verbalized_patterns)):
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
        if index % 100 == 0:
            print(f'{index}/{length}')
        word = get_predicted_word(tokenizer, model, ss)
        pred_y = deverbalize(word)
        pred_ys.append(pred_y)
        true_ys.append(labels[1])
    return pred_ys, true_ys

def cal_scores(pred_ys, true_ys):
    precision, recall, fscore, support = score(true_ys, pred_ys)

def cal_prec_rec_f1_v2(results, targets):
    return f1_score(targets, results, average='macro')

