import json
from sentence_transformers import SentenceTransformer
import torch as t

default_pretrained_model = 'paraphrase-distilroberta-base-v1'
model = None
# model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def load_dataset(path = 'datasets/en_city_train.ds.json'):
  with open(path) as f:
    result = json.load(f)
  return result

def embeddings_from_text(text):
  sentences = text.split('.')
  return sentences
 
def get_longest(sentences):
  max_len = 0
  max_s = None
  max_index = None
  for i,s in enumerate(sentences):
    if len(s) > max_len:
      max_len = len(s)
      max_s = s
      max_index = i
  return max_s, max_index, max_len

def sentence_to_embedding(s):
  global model
  if model is None:
    model = SentenceTransformer(default_pretrained_model)
    print(f'Inited s-bert model using {default_pretrained_model}')
  else: 
    pass
  # dd
  return t.Tensor(model.encode(s))

