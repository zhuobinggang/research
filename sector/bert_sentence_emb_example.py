import torch as t
from transformers import BertModel, BertJapaneseTokenizer

model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
toker = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

ids = toker.encode('吾輩はここで始めて人間というものを見た。')
ids = t.LongTensor([ids])

dic = model(input_ids = ids, return_dict = True)
last_hidden_state = dic['last_hidden_state']
print(last_hidden_state.shape)
cls_embedding = last_hidden_state[0][0] 
print(cls_embedding.shape)

