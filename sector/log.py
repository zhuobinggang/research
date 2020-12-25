import data_operator as data
import torch as t

s_bert_out_size = 768
input_size = 99
hidden_size = 99
batch_size = 1

def log():
  ss = ['SOF', 'hello world!', 'EOF']
  embs = t.stack([data.sentence_to_embedding(s) for s in ss]) # (seq_len, s_bert_out_size)
  minify_layer = t.nn.Linear(s_bert_out_size, input_size)
  minified_embs = minify_layer(embs).view(-1, batch_size, input_size) # (seq_len, batch_size, input_size)
  encoder = t.nn.LSTM(input_size, hidden_size)

  
  
