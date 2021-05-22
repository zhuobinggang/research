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

def encode_without_special_tokens(toker, s, max_len = 240):
  ids = toker.encode(s, add_special_tokens = False)
  return ids[:max_len]

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

def get_left_right_ids_no_special_token(toker, ss, pos, max_len = None):
  if max_len is None:
    max_len = int(500 / len(ss)) # 4句时候125 tokens/句, 2句250 tokens/句
  idss = [encode_without_special_tokens(toker, s, max_len = max_len) for s in ss] # 左右两边不应过长
  left = flatten_num_lists(idss[0: pos])
  right = flatten_num_lists(idss[pos:])
  return left, right

# return: (784)
def compress_one_cls_one_sep_pool_cls(bert, toker, ss, pos):
  left, right = get_left_right_ids_no_special_token(toker, ss, pos)
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
  left, right = get_left_right_ids_no_special_token(toker, ss, pos)
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

def compress_by_ss_pair_get_mean(bert, toker, ss):
  assert len(ss) == 2
  ids = encode_sentence_pair(toker, ss[0], ss[1])
  ids = t.LongTensor(ids).view(1, -1)
  if GPU_OK:
    ids = ids.cuda()
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  out = out.view(length, hidden_size)
  out = out.mean(0)
  assert len(out.shape) == 1 and out.shape[0] == hidden_size
  return out

def compress_by_ss_pos_get_mean(bert, toker, ss, pos):
  idss = [encode_without_special_tokens(toker, s) for s in ss]
  left = flatten_num_lists(idss[0: pos])
  right = flatten_num_lists(idss[pos:])
  ids = add_special_token_for_ids_pair(toker, left, right)
  ids = t.LongTensor(ids).view(1, -1)
  if GPU_OK:
    ids = ids.cuda()
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  assert length == len(left) + len(right) + 2
  out = out.view(length, hidden_size)
  out = out.mean(0)
  assert len(out.shape) == 1 and out.shape[0] == hidden_size
  return out

def get_idss_multiple_seps(toker, ss, pos, max_len = None):
  if max_len is None:
    max_len = int(500 / len(ss)) # 4句时候125 tokens/句, 2句250 tokens/句
  idss = [encode_without_special_tokens(toker, s, max_len = max_len) for s in ss] # 左右两边不应过长
  cls_id = toker.cls_token_id
  sep_id = toker.sep_token_id
  idss = [ids + [sep_id] for ids in idss] # Add [SEP]
  ids = flatten_num_lists(idss)
  ids = [cls_id] + ids
  return ids

def compress_by_ss_get_all_tokens(bert, toker, ss, max_len = None):
  if max_len is None:
    max_len = int(500 / len(ss)) # 4句时候125 tokens/句, 2句250 tokens/句
  idss = [encode_without_special_tokens(toker, s, max_len = max_len) for s in ss] # 左右两边不应过长
  return wrap_idss_with_special_tokens(idss)

# [cls] s1 [sep] s2 [sep] s3 [sep]
# 返回[[cls],[sep],[sep],[sep]]
def compress_by_ss_get_special_tokens(bert, toker, ss, max_len = None):
  cls, seps, sentence_tokens = compress_by_ss_get_all_tokens(bert, toker, ss, max_len)
  return cls, seps

def compress_by_ss_get_cls_and_middle_sep(bert, toker, ss, max_len = None):
  cls, seps = compress_by_ss_get_special_tokens(bert, toker, ss, max_len)
  seps_middle_pos = int(len(ss) / 2) - 1
  return cls, seps[seps_middle_pos]


def wrap_idss_with_special_tokens(bert, toker, idss):
  origin_lengths = [len(ids) for ids in idss]
  cls_id = toker.cls_token_id
  sep_id = toker.sep_token_id
  idss = [ids + [sep_id] for ids in idss] # Add [SEP]
  ids = flatten_num_lists(idss)
  ids = [cls_id] + ids # Add [CLS]
  ids = t.LongTensor(ids).view(1, -1)
  if GPU_OK:
    ids = ids.cuda()
  out = bert(input_ids = ids, return_dict = True)['last_hidden_state']
  batch, length, hidden_size = out.shape
  assert length == sum(origin_lengths) + len(idss) + 1 # The only assertaion
  out = out.view(length, hidden_size)
  cls = out[0]
  out = out[1:] # 剪掉cls
  outs = []
  for l in origin_lengths:
    outs.append(out[:l + 1])
    out = out[l + 1:]
  for o, org_length in zip(outs, origin_lengths):
    assert o.shape[0] == org_length + 1 # 因为带了SEP
  seps = [o[-1] for o in outs]
  sentence_tokens = [o[:-1] for o in outs]
  assert len(seps) == len(idss)
  seps = t.stack(seps)
  assert len(seps.shape) == 2
  return cls, seps, sentence_tokens

def compress_by_ss_then_pad(bert, toker, ss, pos, len2pad, max_len = None):
  if max_len is None:
    max_len = int(500 / len2pad) # 4句时候125 tokens/句, 2句250 tokens/句
  idss = [encode_without_special_tokens(toker, s, max_len = max_len) for s in ss] # 左右两边不应过长
  # pad idss with empty sentences
  pad_left_nums = None
  pad_right_nums = None
  if pos < (len2pad / 2):
    # pad [(len/2) - pos] sentence to left
    pad_left_nums = int((len2pad / 2) - pos)
    for i in range(pad_left_nums):
      idss = [[]] + idss
  elif len2pad != len(ss):
    pad_right_nums = len2pad - len(ss)
    # pad (len2pad - len) sentence to right
    for i in range(len2pad - len(ss)):
      idss = idss + [[]]
  assert len(idss) == len2pad
  cls, seps, sentence_tokens = wrap_idss_with_special_tokens(bert, toker, idss)
  assert len(seps) == len2pad
  if pad_left_nums is not None:
    seps = seps[pad_left_nums:]
  if pad_right_nums is not None:
    seps = seps[0:-pad_right_nums]
  assert len(seps) == len(ss)
  return cls, seps, sentence_tokens


