from transformers import BertModel, BertJapaneseTokenizer
from importlib import reload
import torch as t
GPU_OK = t.cuda.is_available()

model = None
tokenizer = None

def try_init_bert():
  global model, tokenizer
  if model is None:
    model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
  
# return: (batch_size, 768)
def batch_get_embs(ss):
  try_init_bert()
  batch = tokenizer(ss, padding=True, truncation=True, return_tensors="pt")
  input_ids = batch['input_ids']
  attention_mask = batch['attention_mask']
  # See the models docstrings for the detail of the inputs
  outputs = model(input_ids, attention_mask, return_dict=True)
  return outputs.pooler_output # [CLS] : (batch_size, 768) 

def encode_without_special_tokens(toker, s):
  return toker.encode(s, add_special_tokens = False)

def encode_with_special_tokens(toker, s):
  return toker.encode(s)

def add_special_token_for_ids_pair(toker, ids1, ids2):
  cls_id = toker.cls_token_id
  sep_id = toker.sep_token_id
  return [cls_id] + ids1 + [sep_id] + ids2

def encode_sentence_pair(toker, s1, s2):
  ids1 = encode_without_special_tokens(toker, s1)
  ids2 = encode_without_special_tokens(toker, s2)
  return add_special_token_for_ids_pair(toker, ids1, ids2)


# return: (seq_len, 784)
def compress_left_get_embs(bert, toker, left):
  if len(left) == 1:
    ids = encode_with_special_tokens(toker, left[0])
    seq_len = len(ids) - 2
    ids = t.LongTensor(ids)
    ids = ids.view(1, -1) # (batch_size, sequence_length)
    if GPU_OK:
      ids = ids.cuda()
    dic = bert(input_ids = ids, return_dict = True)
    out = dic['last_hidden_state']
    batch, length, hidden_size = out.shape
    out = out.view(length, hidden_size)
    out = out[1:-1]
    assert out.shape[0] == seq_len
    return out
  elif len(left) == 2:
    lids = encode_without_special_tokens(toker, left[0])
    rids = encode_without_special_tokens(toker, left[1])
    rseq_len = len(rids)
    lseq_len = len(lids)
    ids = t.LongTensor(add_special_token_for_ids_pair(toker, lids, rids)).view(1, -1)
    if GPU_OK:
      ids = ids.cuda()
    dic = bert(input_ids = ids, return_dict = True)
    out = dic['last_hidden_state']
    batch, length, hidden_size = out.shape
    assert length == 1 + lseq_len + 1 + rseq_len
    out = out.view(length, hidden_size)
    # 取右边
    out = out[lseq_len + 2:]
    assert out.shape[0] == rseq_len
    return out


# return: (seq_len, 784)
def compress_right_get_embs(bert, toker, right):
  if len(right) == 1:
    ids = encode_with_special_tokens(toker, right[0])
    seq_len = len(ids) - 2
    ids = t.LongTensor(ids)
    ids = ids.view(1, -1) # (batch_size, sequence_length)
    if GPU_OK:
      ids = ids.cuda()
    dic = bert(input_ids = ids, return_dict = True)
    out = dic['last_hidden_state']
    batch, length, hidden_size = out.shape
    out = out.view(length, hidden_size)
    out = out[1:-1]
    assert out.shape[0] == seq_len
    return out
  elif len(right) == 2:
    lids = encode_without_special_tokens(toker, right[0])
    rids = encode_without_special_tokens(toker, right[1])
    rseq_len = len(rids)
    lseq_len = len(lids)
    ids = t.LongTensor(add_special_token_for_ids_pair(toker, lids, rids)).view(1, -1)
    if GPU_OK:
      ids = ids.cuda()
    dic = bert(input_ids = ids, return_dict = True)
    out = dic['last_hidden_state']
    batch, length, hidden_size = out.shape
    assert length == 1 + lseq_len + 1 + rseq_len
    out = out.view(length, hidden_size)
    out = out[1: lseq_len + 1]
    assert out.shape[0] == lseq_len
    return out


