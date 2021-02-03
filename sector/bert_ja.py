from transformers import BertModel, BertJapaneseTokenizer

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
  
