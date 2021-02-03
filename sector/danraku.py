import torch as t
import torch.nn as nn
import torch.nn.functional as F
import data_jap as data
from itertools import chain
import torch.optim as optim
from transformers import BertModel, BertJapaneseTokenizer
import time
import random
import utils as U
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

class Model(nn.Module):
  def __init__(self, verbose = False):
    super().__init__()
    self.verbose = verbose
    self.define_variables()
    self.init_embedding_layer()
    self.init_hook()
    self.init_optim()

  def init_hook(self):
    pass

  def init_embedding_layer(self):
    pass

  def get_should_update(self):
    pass

  def init_optim(self):
    pass

  def define_variables(self):
    self.s_bert_out_size = 768

  def print_info_this_step(self, inpt, target, loss):
    pass

  def get_loss_by_input_and_target(self, outs, labels):
    pass

  def get_embs_from_inpts(self, inpts):
    pass

  def labels_processed(self, labels, _):
    pass

  def get_outs(self, inpts):
    pass

  def get_loss(self, inpts, labels):
    pass

  # inpts: (seq_len, batch_size, m)
  def train(self, inpts, labels):
    if len(inpts) < 1:
      print('Warning: empty training sentence list')
      return None
    loss = self.get_loss(inpts, labels)
    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    return loss.item()


  @t.no_grad()
  def dry_run(self, inpts):
    pass

  # inpts: (seq_len, batch_size, m)
  def forward(self, inpts):
    pass
    


class Model_Con(Model):

  def init_hook(self):
    self.fw = t.nn.Linear(self.s_bert_out_size * 2, self.s_bert_out_size)
    self.fw2 = t.nn.Linear(self.s_bert_out_size, self.s_bert_out_size)
    self.minify = t.nn.Linear(self.s_bert_out_size, 2)
    self.CEL = t.nn.CrossEntropyLoss()

  def init_embedding_layer(self):
    # self.minify_layer = t.nn.Linear(self.s_bert_out_size, self.input_size)
    pass


  def get_should_update(self):
    return chain(self.fw.parameters(), self.minify.parameters(), self.fw2.parameters())

  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update(), 0.001)


  def print_info_this_step(self, inpt, target, loss):
    if self.verbose:
      print(f'The loss: {loss.item()}')

  # labels_processed : (batch_size)
  # outs : (batch_size, 2)
  def get_loss_by_input_and_target(self, outs, labels):
    return self.CEL(outs, labels)
    
  # inpts: (batch_size, (left, right))
  # left/right: [sentence]
  # return: (batch_size, 768 * 2)
  def get_embs_from_inpts(self, inpts):
    results = [] # (batch_size, 1536)
    for left, right in inpts:
      # left_combine = '。'.join(left) # string
      # right_combine = '。'.join(right)
      left_tensor = t.cat([t.from_numpy(data.get_cached_emb(s)) for s in left]) # (768)
      right_tensor = t.cat([t.from_numpy(data.get_cached_emb(s)) for s in right]) # (768)
      combined = t.cat((left_tensor, right_tensor)) # (1536)
      results.append(combined)
    return t.stack(results).detach()

  # labels: (batch_size)
  def labels_processed(self, labels, _):
    # 返回一个规整的tensor(max_seq_len, batch_size, max_seq_len)
    return t.LongTensor(labels) #(batch_size)

  def get_outs(self, inpts):
    embs = self.get_embs_from_inpts(inpts) # (batch_size, 768 * 2)
    outs = self.fw(embs) # (batch_size, 768)
    outs = t.tanh(outs)
    outs = self.fw2(outs)
    outs = t.tanh(outs)
    outs = self.minify(outs) # (batch_size, 2)
    return outs


  def get_loss(self, inpts, labels):
    outs = self.get_outs(inpts) # (batch_size, 2)
    labels_processed = self.labels_processed(labels, None) # (batch_size)
    loss = self.get_loss_by_input_and_target(outs, labels_processed)
    self.print_info_this_step(outs, labels_processed, loss)
    return loss

  # inpts: (seq_len, batch_size, m)
  def train(self, inpts, labels):
    if len(inpts) < 1:
      print('Warning: empty training sentence list')
      return None
    loss = self.get_loss(inpts, labels)
    
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    return loss.item()


  @t.no_grad()
  def dry_run(self, inpts):
    return self.get_outs(inpts).argmax(1).item()

  # inpts: (seq_len, batch_size, m)
  def forward(self, inpts):
    pass
    

class Model_Bert(Model_Con):

  def batch_get_embs(ss):
    try_init_bert()
    batch = tokenizer(ss, padding=True, truncation=True, return_tensors="pt")
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    # See the models docstrings for the detail of the inputs
    outputs = model(input_ids, attention_mask, return_dict=True)
    return outputs.pooler_output # [CLS] : (batch_size, 768) 


  def init_hook(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.fw = t.nn.Linear(self.s_bert_out_size, int(self.s_bert_out_size / 2))
    # self.fw2 = t.nn.Linear(int(self.s_bert_out_size / 2), int(self.s_bert_out_size / 2))
    self.minify = t.nn.Linear(int(self.s_bert_out_size / 2), 2)
    self.CEL = t.nn.CrossEntropyLoss()

  def get_should_update(self):
    return chain(self.fw.parameters(), self.minify.parameters(), self.bert.parameters())

  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update(), 0.001)

  # inpts: (batch_size, (left, right))
  # left/right: [sentence]
  # return: (batch_size, 768)
  def get_embs_from_inpts(self, inpts):
    results = [] # (batch_size, 1536)
    toker = self.tokenizer
    texts = []
    for left_ss, right_ss in inpts:
      the_text = toker.cls_token + toker.sep_token.join(['。'.join(left_ss), '。'.join(right_ss)])
      texts.append(the_text)
    tokenized = toker(texts, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True) # manually add token
    result = self.bert(tokenized.input_ids, tokenized.attention_mask, return_dict=True)
    return result.pooler_output

  def get_loss(self, inpts, labels):
    outs = self.get_outs(inpts) # (batch_size, 2)
    labels_processed = self.labels_processed(labels, None) # (batch_size)
    loss = self.get_loss_by_input_and_target(outs, labels_processed)
    self.print_info_this_step(outs, labels_processed, loss)
    return loss

  def get_outs(self, inpts):
    embs = self.get_embs_from_inpts(inpts) # (batch_size, 768)
    outs = self.fw(embs) # (batch_size, 768)
    outs = t.tanh(outs)
    # outs = self.fw2(outs) # (batch_size, 768)
    # outs = t.tanh(outs)
    outs = self.minify(outs) # (batch_size, 2)
    return outs

