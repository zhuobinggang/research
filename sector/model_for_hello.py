import torch as t
import torch.nn as nn
import torch.nn.functional as F
import data_operator_hello_world as data
from itertools import chain
import torch.optim as optim
import model_bilstm as model
import logging

import utils as U


class Model(model.Model_BCE_Adam):

  def init_logger(self):
    print('inited logger for hello project!!! Output to model_bilstm_hello_world.log')
    logging.basicConfig(
      filename='model_bilstm_hello_world.log',
      format='%(asctime)s %(levelname)-8s %(message)s',
      level=logging.DEBUG,
      datefmt='%Y-%m-%d %H:%M:%S')

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

  def output(self, mat, nums, path='dd.png'):
    output_heatmap(mat, nums, nums, path)

  def dry_run_then_output(self, path='dd.png'):
    nums, att_mat = data.generate_datas()
    nums = list(sorted(nums))
    o = self.dry_run(nums)
    self.output(o, nums, path)


def output_heatmap(mat, xs, ys, path = 'dd.png'):
  U.output_heatmap(mat, xs, ys, path)

def get_train_datas(): 
  return data.read()


class Model_No_Diagonal_Zero(Model):
  def zero_diagonal(self, mat):
    return mat
