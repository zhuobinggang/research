import torch as t
import model
from model import ordered_index, save, load
import torch.optim as optim
from itertools import chain

class Model(model.Model):

  def __init__(self, hidden_state_size = 50):
    super().__init__(hidden_state_size)

  def init_encoder(self):
    print('My own method of init_encoder()')
    print('I have no encoder!')

  def init_optim(self):
    print('My own method of init_optim()')
    should_update = chain(self.decoder.parameters(), self.query_from_dh_layer.parameters(), self.W2.parameters(), self.W1.parameters(), self.V.parameters())
    self.optim = optim.SGD(should_update, lr=0.01, momentum=0.9)

  def get_encoded(self, list_of_num):
    # 转成inpts
    inpts = self._inpt_for_encoder(list_of_num.copy()).detach()
    for_select = t.cat((self._h_c_or_file_symbols(self.EOF), inpts)).detach()
    encoder_out = inpts.sum(0).view(-1, 1, self.input_size) # (seq_len, 1, input_size)
    return for_select, encoder_out



