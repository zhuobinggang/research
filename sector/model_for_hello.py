import torch as t
import torch.nn as nn
import torch.nn.functional as F
import data_operator_hello_world as data
from itertools import chain
import torch.optim as optim
import model_bilstm as model

import utils as U


class Model(model.Model_BCE_Adam):
  def init_embedding_layer(self):
    self.embedding = t.nn.Embedding(120, self.input_size)

  def get_should_update(self):
    return chain(self.encoder.parameters(), self.embedding.parameters())

  def get_embs_from_inpts(self, inpts):
    res = t.stack([self.embedding(t.tensor(inpt)) for inpt in inpts])
    #print(res)
    return res

  def labels_processed(self, labels, inpts):
    return t.FloatTensor(labels)


def output_heatmap(mat, xs, ys, path = 'dd.png'):
  U.output_heatmap(mat, xs, ys, path)

def get_train_datas(): 
  return data.read()
