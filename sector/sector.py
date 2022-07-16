import torch as t
import numpy as np
nn = t.nn
F = t.nn.functional
from transformers import BertJapaneseTokenizer, BertModel
import datetime
from itertools import chain
from dataset_for_sector import read_ld_train, read_ld_test, read_ld_dev

# 分裂sector, 2vs2的时候，同时判断三个分割点
class Sector_2022(nn.Module):
  def __init__(self, learning_rate = 5e-6):
    super().__init__()
    self.learning_rate = learning_rate
    self.bert_size = 768
    self.verbose = False
    self.init_bert()
    self.init_hook()
    self.opter = t.optim.AdamW(self.get_should_update(), self.learning_rate)
    self.cuda()

  def init_bert(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
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


def focal_aux_loss(out, labels, fl_rate, aux_rate, main_pos = 1):
    assert len(labels.shape) == 1
    assert len(out.shape) == 1
    assert labels.shape[0] == out.shape[0]
    total = []
    for idx, (o, l) in enumerate(zip(out, labels)):
      pt = o if (l == 1) else (1 - o)
      loss = (-1) * t.log(pt) * t.pow((1 - pt), fl_rate)
      if idx != main_pos: # AUX loss, windows size = 4
          loss = loss * aux_rate
      total.append(loss)
    total = t.stack(total)
    return total.sum()

def focal_loss(o, l, fl_rate):
    assert len(l.shape) == 0 
    assert len(o.shape) == 0 
    pt = o if (l == 1) else (1 - o)
    loss = (-1) * t.log(pt) * t.pow((1 - pt), fl_rate)
    return loss
    
def train(m, ds_train, epoch = 1, batch = 16, fl_rate = 0, aux_rate = 0):
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = m.opter
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        for row_idx, (ss, labels) in enumerate(np.random.permutation(ds_train)):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            combined_ids, sep_idxs = encode(ss, toker)
            labels = [int(label) for label in labels]
            labels = labels[1:] # (3)，不需要第一个label
            sep_idxs = sep_idxs[0:-1] # (3), 不需要最后一个sep
            out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, sep_idxs, :] # (1, 3, 768)
            out_mlp = m.classifier(out_bert) # (1, 3, 1)
            # cal loss
            loss = focal_aux_loss( # 不使用focal loss & 不使用辅助损失
                    out_mlp.squeeze(), 
                    t.LongTensor(labels).cuda(), 
                    fl_rate = fl_rate, 
                    aux_rate = aux_rate,
                    main_pos = 1)
            loss.backward()
            # backward
            if (row_idx + 1) % batch == 0:
                opter.step()
                opter.zero_grad()
    opter.step()
    opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)
    return delta.seconds
  
# NOTE: 还需要更改loss function以抹除aux loss的tricky操作
def train_baseline(m, ds_train, epoch = 1, batch = 16, fl_rate = 0):
    first_time = datetime.datetime.now()
    toker = m.toker
    bert = m.bert
    opter = m.opter
    CLS_POS = 0 # baseline需要调用encode_standard（只添加一个sep在中间）然后取出CLS对应的embedding
    for epoch_idx in range(epoch):
        print(f'Train epoch {epoch_idx}')
        for row_idx, (ss, labels) in enumerate(np.random.permutation(ds_train)):
            if row_idx % 1000 == 0:
                print(f'finished: {row_idx}/{len(ds_train)}')
                pass
            combined_ids, _ = encode_standard(ss, toker)
            labels = [int(label) for label in labels]
            assert len(labels) == 4
            label = labels[2] # 取中间的label
            out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, CLS_POS, :] # (1, 1, 768)
            out_mlp = m.classifier(out_bert) # (1, 1, 1)
            # cal loss
            loss = focal_loss( # 不使用focal loss & 不使用辅助损失
                    out_mlp.squeeze(), # (scalar)
                    t.tensor(label).cuda(), # (scalar)
                    fl_rate = fl_rate)
            loss.backward()
            # backward
            if (row_idx + 1) % batch == 0:
                opter.step()
                opter.zero_grad()
    opter.step()
    opter.zero_grad()
    last_time = datetime.datetime.now()
    delta = last_time - first_time
    print(delta.seconds)
    return delta.seconds

# 复数个SEP
# NOTE: 更改了encode之后train的逻辑也要更改
def encode(ss, toker):
    PART_LEN_MAX = int(500 / len(ss)) # 默认是512上限，考虑到特殊字符使用500作为分子
    idss = []
    for s in ss:
        ids = toker.encode(s, add_special_tokens = False)
        if len(ids) > PART_LEN_MAX:
            # print(f'WARN: len(ids) > PART_LEN_MAX! ===\n{s}')
            idss.append(ids[:PART_LEN_MAX])
        else:
            idss.append(ids)
    combined_ids = [toker.cls_token_id]
    sep_idxs = []
    idx_counter = 1
    for ids in idss:
        ids.append(toker.sep_token_id)
        idx_counter += len(ids)
        sep_idxs.append(idx_counter - 1)
        combined_ids += ids
    return t.LongTensor(combined_ids), sep_idxs

# 单个SEP
def encode_standard(ss, toker):
    PART_LEN_MAX = int(500 / len(ss)) # 默认是512上限，考虑到特殊字符使用500作为分子
    idss = []
    for s in ss:
        ids = toker.encode(s, add_special_tokens = False)
        if len(ids) > PART_LEN_MAX:
            # print(f'WARN: len(ids) > PART_LEN_MAX! ===\n{s}')
            idss.append(ids[:PART_LEN_MAX])
        else:
            idss.append(ids)
    combined_ids = [toker.cls_token_id]
    sep_idxs = []
    assert len(idss) == 4
    # left
    for i in range(0, 2):
        ids = idss[i]
        combined_ids += ids
    sep_idxs.append(len(combined_ids))
    combined_ids.append(toker.sep_token_id)
    # right
    for i in range(2, 4):
        ids = idss[i]
        combined_ids += ids
    return t.LongTensor(combined_ids), sep_idxs

def test(ds_test, m):
    y_true = []
    y_pred = []
    toker = m.toker
    bert = m.bert
    for row_idx, row in enumerate(ds_test):
        ss, labels = row
        combined_ids, sep_idxs = encode(ss, toker)
        labels = labels[1:] # (3)，不需要第一个label
        sep_idxs = sep_idxs[0:-1] # (3), 不需要最后一个sep
        out_bert = bert(combined_ids.unsqueeze(0).cuda()).last_hidden_state[:, sep_idxs, :] # (1, 3, 768)
        out_mlp = m.classifier(out_bert) # (1, 3, 1)
        y_pred += out_mlp.squeeze().tolist()
        y_true += labels
    return y_true, y_pred

def fomatted_results(y_true, y_pred):
    MAX_LEN = len(y_pred)
    trues = []
    preds = []
    preds_rounded = []
    idx = 1
    while idx < MAX_LEN:
        trues.append(y_true[idx])
        preds.append(y_pred[idx])
        preds_rounded.append(1 if y_pred[idx] > 0.5 else 0)
        idx += 3
    return trues, preds, preds_rounded

def test_chain(m, ld_test):
    y_true, y_pred = test(ld_test, m)
    trues, _, preds_rounded = fomatted_results(y_true, y_pred)
    return cal_prec_rec_f1_v2(preds_rounded, trues)

def cal_prec_rec_f1_v2(results, targets):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  for guess, target in zip(results, targets):
    if guess == 1:
      if target == 1:
        TP += 1
      elif target == 0:
        FP += 1
    elif guess == 0:
      if target == 1:
        FN += 1
      elif target == 0:
        TN += 1
  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  rec = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
  balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
  balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
  return prec, rec, f1, balanced_acc
