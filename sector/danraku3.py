import danraku2 as model
import torch as t
from transformers import BertModel, BertJapaneseTokenizer

# inpts: (batch_size, (left, right))
# return: (batch_size, 768 * 2)
def get_embs_from_inpts(self, inpts):
  batches = [] # (batch_size * 2, sentence)
  for left_ss, right_ss in inpts:
    left_sentence = '。'.join(left_ss)
    right_sentence = '。'.join(right_ss)
    # left = self.tokenizer(left_sentence, add_special_tokens=True, return_tensors='pt', truncation=True)
    batches.append(left_sentence)
    batches.append(right_sentence)
  tokenized = self.tokenizer(batches, add_special_tokens=True, return_tensors='pt', padding=True, truncation=True) # manually add token
  batch_embeddings = self.bert(tokenized.input_ids, tokenized.attention_mask, return_dict=True).pooler_output # (batch_size * 2, 768)
  return batch_embeddings.view(len(inpts), 2 * 768)

class BERT_Cat_Sentence(model.Model_Bert_Balanced_CE):
  def init_hook(self):
    self.bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.bert.train()
    self.tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    self.fw = t.nn.Linear(self.s_bert_out_size * 2, self.s_bert_out_size)
    self.minify = t.nn.Linear(self.s_bert_out_size, 1)

  def get_outs(self, inpts):
    embs = get_embs_from_inpts(self, inpts) # (batch_size, 768 * 2)
    outs = self.fw(embs) # (batch_size, 768)
    outs = self.minify(outs) # (batch_size, 1)
    return outs

