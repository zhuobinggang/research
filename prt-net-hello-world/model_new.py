import torch as t
nn = t.nn
torch = t
import numpy as np
from importlib import reload
import torch.optim as optim
from itertools import chain
import logging

GPU_OK = t.cuda.is_available()

# ============= Position Encoding ===============

def position_encoding_ddd(t, i, d):
  k = int(i/2)
  omiga = 1 / np.power(10000, 2 * k / d)
  even = (i / 2).is_integer()
  return np.sin(omiga * t) if even else np.cos(omiga * t)

def cuda(emb):
  return emb.cuda() if GPU_OK else emb

# seq: (seq_len, feature)
# return: (seq_len, feature)
def position_encoding(seq):
  embs = []
  for t, data in enumerate(seq):
    d = data.shape[0]
    pos_emb = [position_encoding_ddd(t, i, d) for i in range(0, d)]
    pos_emb = torch.tensor(pos_emb)
    embs.append(pos_emb)
  embs = torch.stack(embs)
  return cuda(embs)

# seq: (seq_len, feature)
# return: (seq_len, feature)
def position_encoding_(seq):
  for t, data in enumerate(seq):
    d = data.shape[0]
    pos_emb = [position_encoding_ddd(t, i, d) for i in range(0, d)]
    pos_emb = torch.tensor(pos_emb)
    data = data + pos_emb

# ============= Model ===============

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    feature = 64
    self.feature = feature
    self.embedding_layer = nn.Embedding(102, feature)
    self.id_EOS = t.LongTensor([100])
    self.id_SOF = t.LongTensor([101])
    self.init_layers()
    self.gru = nn.GRU(self.feature, self.feature, batch_first=True, bidirectional=False)
    self.CEL = nn.CrossEntropyLoss()
    self.optim = optim.AdamW(self.get_should_update(), 1e-3)
    self.verbose = False

  def init_layers(self):
    self.selfatt_layer = nn.TransformerEncoderLayer(d_model=feature, nhead=16, dim_feedforward = int(feature * 1.5), dropout = 0)

  def get_should_update(self):
    return chain(self.embedding_layer.parameters(), self.selfatt_layer.parameters(), self.gru.parameters())

  # inpts: tensor([6, 9, 8])
  def ember(self, inpts):
    return self.embedding_layer(inpts)

  # embs: (seq_len, feature)
  def add_EOS(self, embs):
    eos = self.embedding_layer(self.id_EOS) # (1, feature)
    return t.cat((embs, eos))

  # (seq_len + 1, feature)
  def add_pos_encoding(self, embs):
    # position_encoding_(embs)
    return embs

  # (seq_len + 1, feature)
  # return: (seq_len + 1, feature)
  def selfatt(self, embs):
    seq_len = embs.shape[0]
    feature = embs.shape[1]
    return self.selfatt_layer(embs.view(1, seq_len, feature)).view(seq_len, feature)

  # decoder_input: (1, feature)
  def seq_output(self, decoder_input):
    _, h_next = self.gru(decoder_input.view(1, decoder_input.shape[0], decoder_input.shape[1]))
    return h_next.view(decoder_input.shape[0], decoder_input.shape[1])

  # query: (1, feature)
  # embs: (seq_len + 1, feature)
  # return: (1, seq_len + 1)
  def point(self, query, embs):
    return t.mm(embs, query.transpose(0, 1)).view(1, -1)

  def train_step(self, loss):
    self.zero_grad()
    loss.backward()
    self.optim.step()

  def print_info(self, outputs, targets):
    if self.verbose:
      print(f'targets: {targets}, outputs: {outputs}')

  def get_sof(self):
    return self.embedding_layer(self.id_SOF)

  # inpts: tensor([6, 9, 8])
  # labels: tensor([0, 2, 1, 3])
  def train(self, inpts, labels): 
    embs = self.ember(inpts) # (seq_len, feature)
    embs = self.add_EOS(embs) # (seq_len + 1, feature)
    embs = self.add_pos_encoding(embs)
    embs = self.selfatt(embs) # (seq_len + 1, feature)
    # emb_EOS = embs[-1] # (feature)
    decoder_input = self.get_sof() # (1, feature)
    result_losss = []
    outputs = []
    targets = []
    loss = torch.tensor([0.0])
    for label in labels:
      decoder_output = self.seq_output(decoder_input) # (1, feature)
      pointed_output = self.point(decoder_output, embs) # (1, seq_len + 1)
      result_id = pointed_output.argmax()
      true_next_id = label.item() # Correct input feed in
      outputs.append(result_id.item())
      targets.append(label.item())
      loss += self.CEL(pointed_output, label.view(1))
      result_losss.append(loss.detach().item())
      decoder_input = embs[result_id].view(1, -1)
    self.train_step(loss)
    self.print_info(outputs, targets)
    return np.average(result_losss)

  @t.no_grad()
  def dry_run(self, inpts, labels):
    embs = self.ember(inpts) # (seq_len, feature)
    embs = self.add_EOS(embs) # (seq_len + 1, feature)
    embs = self.add_pos_encoding(embs)
    embs = self.selfatt(embs) # (seq_len + 1, feature)
    # emb_EOS = embs[-1] # (feature)
    decoder_input = self.get_sof() # (1, feature)
    outputs = []
    targets = []
    for label in labels:
      decoder_output = self.seq_output(decoder_input) # (1, feature)
      pointed_output = self.point(decoder_output, embs) # (1, seq_len + 1)
      result_id = pointed_output.argmax()
      outputs.append(result_id.item())
      targets.append(label.item())
      decoder_input = embs[result_id].view(1, -1)
    self.print_info(outputs, targets)
    return outputs, targets


class Model_Pos_Encoding(Model):
  # (seq_len + 1, feature)
  def add_pos_encoding(self, embs):
    position_encoding_(embs)
    return embs


class Model_Emb_Encoder(Model):
  # inpts: tensor([6, 9, 8])
  # labels: tensor([0, 2, 1, 3])
  def train(self, inpts, labels): 
    embs = self.ember(inpts) # (seq_len, feature)
    embs = self.add_EOS(embs) # (seq_len + 1, feature)
    # emb_EOS = embs[-1] # (feature)
    decoder_input = self.get_sof() # (1, feature)
    result_losss = []
    outputs = []
    targets = []
    loss = torch.tensor([0.0])
    for label in labels:
      decoder_output = self.seq_output(decoder_input) # (1, feature)
      pointed_output = self.point(decoder_output, embs) # (1, seq_len + 1)
      result_id = pointed_output.argmax()
      true_next_id = label.item() # Correct input feed in
      outputs.append(result_id.item())
      targets.append(label.item())
      loss += self.CEL(pointed_output, label.view(1))
      result_losss.append(loss.detach().item())
      decoder_input = embs[result_id].view(1, -1)
    self.train_step(loss)
    self.print_info(outputs, targets)
    return np.average(result_losss)

  @t.no_grad()
  def dry_run(self, inpts, labels):
    embs = self.ember(inpts) # (seq_len, feature)
    embs = self.add_EOS(embs) # (seq_len + 1, feature)
    # emb_EOS = embs[-1] # (feature)
    decoder_input = self.get_sof() # (1, feature)
    outputs = []
    targets = []
    for label in labels:
      decoder_output = self.seq_output(decoder_input) # (1, feature)
      pointed_output = self.point(decoder_output, embs) # (1, seq_len + 1)
      result_id = pointed_output.argmax()
      outputs.append(result_id.item())
      targets.append(label.item())
      decoder_input = embs[result_id].view(1, -1)
    self.print_info(outputs, targets)
    return outputs, targets

Model_Default = Model_Emb_Encoder


class Model_TF_Decoder(Model_Default):
  # inpts: tensor([6, 9, 8])
  # labels: tensor([0, 2, 1, 3])
  def train(self, inpts, labels): 
    embs = self.ember(inpts) # (seq_len, feature)
    embs = self.add_EOS(embs) # (seq_len + 1, feature)
    # emb_EOS = embs[-1] # (feature)
    decoder_inputs = [self.get_sof()] # (?, feature)
    result_losss = []
    outputs = []
    targets = []
    loss = torch.tensor([0.0])
    for label in labels:
      decoder_output = self.seq_output(decoder_inputs) # (1, feature)
      pointed_output = self.point(decoder_output, embs) # (1, seq_len + 1)
      result_id = pointed_output.argmax()
      true_next_id = label.item() # Correct input feed in
      outputs.append(result_id.item())
      targets.append(label.item())
      loss += self.CEL(pointed_output, label.view(1))
      result_losss.append(loss.detach().item())
      decoder_input = embs[result_id].view(1, -1)
      decoder_inputs.append(decoder_input)
    self.train_step(loss)
    self.print_info(outputs, targets)
    return np.average(result_losss)

  @t.no_grad()
  def dry_run(self, inpts, labels):
    embs = self.ember(inpts) # (seq_len, feature)
    embs = self.add_EOS(embs) # (seq_len + 1, feature)
    # emb_EOS = embs[-1] # (feature)
    decoder_inputs = [self.get_sof()] # (?, feature)
    outputs = []
    targets = []
    for label in labels:
      decoder_output = self.seq_output(decoder_inputs) # (1, feature)
      pointed_output = self.point(decoder_output, embs) # (1, seq_len + 1)
      result_id = pointed_output.argmax()
      outputs.append(result_id.item())
      targets.append(label.item())
      decoder_input = embs[result_id].view(1, -1)
      decoder_inputs.append(decoder_input)
    self.print_info(outputs, targets)
    return outputs, targets

  # decoder_inputs: (n, 1, feature)
  # return: (1, feature)
  def seq_output(self, decoder_inputs):
    decoder_inputs = t.stack(decoder_inputs) # decoder_inputs: (n, 1, feature)
    seq_len, batch, feature = decoder_inputs.shape
    out = self.selfatt_layer(decoder_inputs)
    return out[-1]


class Model_TF_With_Pos(Model_TF_Decoder):
  # decoder_inputs: (n, 1, feature)
  # return: (1, feature)
  def seq_output(self, decoder_inputs):
    decoder_inputs = t.stack(decoder_inputs) # decoder_inputs: (n, 1, feature)
    seq_len, batch, feature = decoder_inputs.shape
    decoder_inputs = decoder_inputs.view(seq_len, feature)
    pos_codings = position_encoding(decoder_inputs) # (n, feature)
    decoder_inputs = decoder_inputs + pos_codings # (n, feature)
    decoder_inputs = decoder_inputs.view(seq_len, 1, feature) # (n, 1, feature)
    out = self.selfatt_layer(decoder_inputs.float())
    return out[-1]
