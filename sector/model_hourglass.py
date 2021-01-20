import torch as t
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import torch.optim as optim
import time
import random
import utils as U
import model_bilstm as model

class Model_No_Diagonal_Zero(model.Model_BCE_Adam):

  def zero_diagonal(self, mat):
    return mat

class Model(Model_No_Diagonal_Zero):
  def get_should_update(self):
    return chain(self.encoder.parameters(), self.minify_layer.parameters(), self.left_lstm.parameters(), self.right_lstm.parameters(), self.squeezed_layer.parameters(), self.query_layer.parameters())

  def init_hook(self):
    self.left_lstm = t.nn.LSTM(self.hidden_size, self.hidden_size) # Bi-Direction
    self.right_lstm = t.nn.LSTM(self.hidden_size, self.hidden_size) # Bi-Direction
    self.query_layer = t.nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.squeezed_layer = t.nn.Linear(self.hidden_size, 1)
    self.BCE = t.nn.BCELoss(None, None, None, 'sum')

  def zero_input(self):
    return t.zeros(1,1,self.input_size)

  def get_scores(self, outs, embs):
    left_lstm = self.left_lstm
    right_lstm = self.right_lstm
    scores = []
    for index,query in enumerate(outs):
      left = embs[0: index] # (left_count, 1, input_size) && reversed
      right = embs[index:] # (right_count, 1, input_size)
      query = self.query_layer(query).view(1,1,-1)
      if len(left) > 0:
        left_out,_ = left_lstm(left.flip(0), (query, self.zero_input())) #  (left_count, 1, input_size)
        left_out = left_out.flip(0)
        left_squeezed = self.squeezed_layer(left_out) # (left_count, 1, 1)
        left_sigmoided = self.sigmoid(left_squeezed)
        # total_loss += self.multiple_pointer_loss(squeezed, left_label)
      if len(right) > 0:
        right_out,_ = right_lstm(right, (query, self.zero_input())) #  (left_count, 1, input_size)
        right_squeezed = self.squeezed_layer(right_out) # (left_count, 1, 1)
        # total_loss += self.multiple_pointer_loss(squeezed, right_out)
        right_sigmoided = self.sigmoid(right_squeezed)
      if len(left) > 0 and len(right) > 0:
        all_sigmoided = t.cat((left_sigmoided, right_sigmoided))
      elif len(left) > 0:
        all_sigmoided = left_sigmoided
      else:
        all_sigmoided = right_sigmoided
      scores.append(all_sigmoided)
    scores = t.cat(scores).view(len(embs), len(embs)) # (seq_len, seq_len)
    return scores


