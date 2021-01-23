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
  return [sentence_to_embedding(s) for s in sentences]
 
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

def ss_to_embs(ss):
  global model
  if model is None:
    model = SentenceTransformer(default_pretrained_model)
    print(f'Inited s-bert model using {default_pretrained_model}')
  else: 
    pass
  # dd
  return t.Tensor(model.encode(ss))

def result_sentences_and_indexs_and_section_num(row):
  annotations = row['annotations']
  text = row['text']
  # sentences = [s.rstrip() for s in text.split('.')]
  # sections =  [text[a['begin'] : a['begin'] + a['length']].replace('\r', '').replace('\n', '') for a in annotations]
  sections =  [text[a['begin'] : a['begin'] + a['length']] for a in annotations]
  sss = [[s for s in sec.split('\n') if s != ''] for sec in sections]
  indexs = []
  result_sentences = []
  acc_index = 0 # 考虑0
  for ss in sss:
    acc_index += len(ss)
    result_sentences += ss
    indexs.append(acc_index)
  indexs.pop()
  return result_sentences, indexs, len(sections)


# 处理wikisection dataset, 调用read data函数，输入文件名，输出[(sentences,correct indexs)]
def read_data(path = 'datasets/en_city_train.ds.json'):
  rows = load_dataset(path)
  return [result_sentences_and_indexs_and_section_num(row)for row in rows]

def read_trains():
  return read_data('datasets/en_city_train.ds.json')
  
def read_tests():
  return read_data('datasets/en_city_test.ds.json')

def text2paragraphs(text):
  return [s for s in text.split('\n') if len(s) > 0]


def text2paragraphs(text):
  return [s for s in text.split('\n') if len(s) > 0]

def text2sentences(text):
  return [s for s in text.replace('\r', '').replace('\n', '').split('.') if len(s) > 0]
