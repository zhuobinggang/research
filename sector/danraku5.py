import danraku2 as model
import torch as t
from transformers import BertModel, BertTokenizer
from pyknp import Juman

SPACE = ' '

class JumanTokenizer():
  def __init__(self):
    self.juman = Juman()
  
  def tokenize(self, text):
    result = self.juman.analysis(text)
    return [mrph.midasi for mrph in result.mrph_list()]


def combine_left_right(self, inpts):
  toker = self.tokenizer
  texts = []
  for left_ss, right_ss in inpts:
    left = '。'.join(left_ss)
    right = '。'.join(right_ss)
    try:
      the_text = toker.cls_token + SPACE + toker.sep_token.join([SPACE.join(self.juman.tokenize(left)) + SPACE, SPACE + SPACE.join(self.juman.tokenize(right))])
    except Exception as e:
      print(f'ERRRRRRROR OCCUR!!! {type(e)}')
      the_text = toker.cls_token + SPACE + toker.sep_token
      print(f'left_ss: {left_ss}, right_ss: {right_ss}, but return: {the_text}')
    texts.append(the_text)
  return texts

def get_embs_from_inpts(self, inpts):
  texts = combine_left_right(self, inpts)
  tokenized = self.tokenizer(texts, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True) # manually add token
  return self.bert(tokenized.input_ids, tokenized.attention_mask, return_dict=True).pooler_output

# 黑桥研究室bert, Juman++
class BERT_Kuro(model.Model_Bert_Balanced_CE):
  def init_hook(self):
    self.bert = BertModel.from_pretrained("/home/taku/projects/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers")
    self.bert.train()
    self.juman = JumanTokenizer()
    self.tokenizer = BertTokenizer("/home/taku/projects/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
    self.fw = t.nn.Linear(self.s_bert_out_size, int(self.s_bert_out_size / 2))
    self.minify = t.nn.Linear(int(self.s_bert_out_size / 2), 1)
    self.weight_one = 3
    self.weight_zero = 1

  def get_outs(self, inpts):
    embs = get_embs_from_inpts(self, inpts) # (batch_size, 768)
    outs = self.fw(embs) # (batch_size, 768 / 2)
    outs = self.minify(outs) # (batch_size, 1)
    return outs

# 黑桥研究室bert, Juman++, cat sentence
class Kuro_Catsentence(model.Model_Bert_Balanced_CE):

  def get_batches_processed(self, inpts):
    batches = [] # (batch_size * 2, sentence)
    for left_ss, right_ss in inpts:
      left = '。'.join(left_ss)
      right = '。'.join(right_ss)
      batches.append(SPACE.join(self.juman.tokenize(left)))
      batches.append(SPACE.join(self.juman.tokenize(right)))
    return batches

  # inpts: (batch_size, (left, right))
  # return: (batch_size, 768 * 2)
  def get_embs_from_inpts(self, inpts):
    batches = self.get_batches_processed(inpts)
    tokenized = self.tokenizer(batches, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True) # manually add token
    batch_embeddings = self.bert(tokenized.input_ids, tokenized.attention_mask, return_dict=True).pooler_output # (batch_size * 2, 768)
    return batch_embeddings.view(len(inpts), 2 * 768)

  def init_hook(self):
    self.bert = BertModel.from_pretrained("/home/taku/projects/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers")
    self.bert.train()
    self.juman = JumanTokenizer()
    self.tokenizer = BertTokenizer("/home/taku/projects/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
    self.fw = t.nn.Linear(self.s_bert_out_size * 2, self.s_bert_out_size)
    self.minify = t.nn.Linear(self.s_bert_out_size, 1)
    self.weight_one = 3
    self.weight_zero = 1

  def get_outs(self, inpts):
    embs = self.get_embs_from_inpts(inpts) # (batch_size, 768 * 2)
    outs = self.fw(embs) # (batch_size, 768)
    outs = self.minify(outs) # (batch_size, 1)
    return outs

