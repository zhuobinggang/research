import danraku as model
import torch as t
from transformers import BertModel, BertJapaneseTokenizer
import torch.optim as optim

one_count = 3897
all_count = 28513
zero_count = all_count - one_count
weight_one = all_count / one_count
weight_zero = all_count / zero_count

# outs : (batch_size, 1)
# labels_processed : (batch_size)
def cal_balanced_loss(xs, ys):
  sigmoided = t.sigmoid(xs).view(-1)
  return ((-1) * ((weight_one * ys * t.log(sigmoided)) + (weight_zero * (1 - ys) * t.log(1 - sigmoided)))).mean()

# inpts: (batch_size, (left, right))
# return: (batch_size, string)
def combine_left_right(self, inpts):
  toker = self.tokenizer
  texts = []
  for left_ss, right_ss in inpts:
    the_text = toker.cls_token + toker.sep_token.join(['。'.join(left_ss), '。'.join(right_ss)])
    texts.append(the_text)
  return texts

def get_embs_from_inpts(self, inpts):
  texts = combine_left_right(self, inpts)
  tokenized = self.tokenizer(texts, add_special_tokens=False, return_tensors='pt', padding=True, truncation=True) # manually add token
  return self.bert(tokenized.input_ids, tokenized.attention_mask, return_dict=True).pooler_output

# =======

class Model_Bert_Balanced_CE(model.Model_Bert):
  def init_hook(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.fw = t.nn.Linear(self.s_bert_out_size, int(self.s_bert_out_size / 2))
    self.minify = t.nn.Linear(int(self.s_bert_out_size / 2), 1)
    # Balanced loss
    self.one_count = 3897
    self.all_count = 28513
    self.zero_count = self.all_count - self.one_count
    self.weight_one = self.all_count / self.one_count 
    self.weight_zero = self.all_count / self.zero_count 

  def get_outs(self, inpts):
    embs = get_embs_from_inpts(self, inpts) # (batch_size, 768)
    outs = self.fw(embs) # (batch_size, 768 / 2)
    outs = self.minify(outs) # (batch_size, 1)
    return outs

  # labels_processed : (batch_size)
  # outs : (batch_size, 1)
  def get_loss_by_input_and_target(self, outs, labels):
    return cal_balanced_loss(outs, labels)

  @t.no_grad()
  def dry_run(self, inpts):
    return 1 if t.sigmoid(self.get_outs(inpts)).item() > 0.5 else 0

  def init_optim(self):
    self.optim = optim.AdamW(self.get_should_update(), 2e-5)


# =======

# =======

class Model_Bert_FL(model.Model_Bert):
  def init_hook(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.fw = t.nn.Linear(self.s_bert_out_size, int(self.s_bert_out_size / 2))
    # self.fw2 = t.nn.Linear(int(self.s_bert_out_size / 2), int(self.s_bert_out_size / 2))
    self.minify = t.nn.Linear(int(self.s_bert_out_size / 2), 1)
    # Balanced loss
    self.one_count = 3897
    self.all_count = 28513
    self.zero_count = self.all_count - self.one_count
    self.weight_one = self.all_count / self.one_count 
    self.weight_zero = self.all_count / self.zero_count 

  def get_outs(self, inpts):
    embs = self.get_embs_from_inpts(inpts) # (batch_size, 768)
    outs = self.fw(embs) # (batch_size, 768 / 2)
    # outs = t.tanh(outs)
    # outs = self.fw2(outs) # (batch_size, 768)
    # outs = t.tanh(outs)
    outs = self.minify(outs) # (batch_size, 1)
    outs = t.sigmoid(outs)
    return outs

  
  # labels_processed : (batch_size)
  # outs : (batch_size, 1) # sigmoid
  def get_loss_by_input_and_target(self, outs, labels):
    losss = []
    for i, row in enumerate(outs): # (1)
      if labels[i].item() == 0:
        loss = (-1) * t.pow(row, 2) * t.log(1-row)
        losss.append(self.weight_zero * loss)
      elif labels[i].item() == 1:
        loss = (-1) * t.pow((1 - row), 2) * t.log(row)
        losss.append(self.weight_one * loss)
    return t.stack(losss).mean()

  @t.no_grad()
  def dry_run(self, inpts):
    return 0 if self.get_outs(inpts).item() < 0.5 else 1

  def init_optim(self):
    self.optim = optim.Adam(self.get_should_update(), 0.001)


