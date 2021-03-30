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

def flatten_num_lists(num_lists):
  return [item for lst in num_lists for item in lst]

# return: (?, 784)
def compress_by_ss_pos_get_emb(bert, toker, ss, pos):
  idss = [encode_without_special_tokens(toker, s) for s in ss]
  target = idss[pos]
  left = flatten_num_lists(idss[pos - 1: pos])
  right = flatten_num_lists(idss[pos:])
  ids = add_special_token_for_ids_pair(toker, left, right)
  ids = t.LongTensor(ids).view(1, -1)
  if GPU_OK:
    ids = ids.cuda()
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  assert length == len(left) + len(right) + 2
  out = out.view(length, hidden_size)
  out_right = out[len(left) + 2:] # 去掉cls, left, sep
  assert out_right.shape[0] == len(right)
  out_target = out_right[0: len(target)]
  assert out_target.shape[0] == len(target)
  return out_target

# return: (784)
def compress_by_ss_pos_get_cls(bert, toker, ss, pos):
  idss = [encode_without_special_tokens(toker, s) for s in ss]
  target = idss[pos]
  left = flatten_num_lists(idss[pos - 1: pos])
  right = flatten_num_lists(idss[pos:])
  ids = add_special_token_for_ids_pair(toker, left, right)
  ids = t.LongTensor(ids).view(1, -1)
  if GPU_OK:
    ids = ids.cuda()
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  assert length == len(left) + len(right) + 2
  out = out.view(length, hidden_size)
  return out[0] # (784)

def compress_by_ss_pos_get_sep(bert, toker, ss, pos):
  idss = [encode_without_special_tokens(toker, s) for s in ss]
  target = idss[pos]
  left = flatten_num_lists(idss[pos - 1: pos])
  right = flatten_num_lists(idss[pos:])
  ids = add_special_token_for_ids_pair(toker, left, right)
  ids = t.LongTensor(ids).view(1, -1)
  if GPU_OK:
    ids = ids.cuda()
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  assert length == len(left) + len(right) + 2
  out = out.view(length, hidden_size)
  out = out[1+len(left):] # [SEP] + right
  assert out.shape[0] == 1 + len(right)
  return out[0] # (784)

def compress_by_ss_pos_get_mean(bert, toker, ss):
  assert len(ss) = 2
  ids = encode_sentence_pair(toker, ss[0], ss[1])
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  out = out.view(length, hidden_size)
  out = out.mean(0)
  assert len(out.shape) == 1 and out.shape[0] == 784
  return out




