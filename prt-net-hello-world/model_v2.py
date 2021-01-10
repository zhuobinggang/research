import model
import torch as t
from my_transformer import Encoder
from itertools import chain
import torch.optim as optim

class Model_AutoReverse_SelfAttend(model.Model_Default):
  def init_optim(self):
    should_update = chain(self.self_attention_layer.parameters(), self.decoder.parameters(), self.W2.parameters(), self.W1.parameters(), self.V.parameters(), self.embedding.parameters())
    self.optim = optim.SGD(should_update, lr=0.01, momentum=0.9)

  def init_encoder(self):
    self.self_attention_layer = Encoder(self.input_size, self.input_size)

  def get_encoded(self, list_of_num):
    # 转成inpts
    inpts = self._inpt_for_encoder(list_of_num.copy()) # (seq_len, 1, input_size)
    inpts_self_attended = self.self_attention_layer(inpts.view(-1, self.input_size)) # (seq_len, input_size)
    for_select = t.cat((self.get_embedding(self.EOF), inpts_self_attended.view(-1, 1, self.input_size)))
    # encoder_out = inpts_self_attended.mean(0).view(1,1,-1) # (input_size) -> (1, 1, input_size)
    return for_select, self.get_embedding(self.DECODER_H0)

class Model_AutoReverse_SelfAttend_V2(Model_AutoReverse_SelfAttend):
  def init_encoder(self):
    self.self_attention_layer = t.nn.TransformerEncoderLayer(self.input_size, 1)

  def get_encoded(self, list_of_num):
    # 转成inpts
    inpts = self._inpt_for_encoder(list_of_num.copy()) # (seq_len, 1, input_size)
    inpts_self_attended = self.self_attention_layer(inpts) # (seq_len, input_size)
    for_select = t.cat((self.get_embedding(self.EOF), inpts_self_attended))
    # encoder_out = inpts_self_attended.mean(0).view(1,1,-1) # (input_size) -> (1, 1, input_size)
    return for_select, self.get_embedding(self.DECODER_H0)
