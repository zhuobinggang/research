from model_new import *

class Self_Att(nn.Module):
  def __init__(self, feature):
    super().__init__()
    self.feature = feature
    self.squre_dk = np.sqrt(self.feature)
    self.WQ = nn.Linear(feature, feature)
    self.WK = nn.Linear(feature, feature)
    self.WV = nn.Linear(feature, feature)

  # embs: (seq_len, feature)
  # return: (seq_len, feature)
  def forward(self, embs):
    qs = self.WQ(embs) # (seq_len, feature)
    ks = self.WK(embs) # (seq_len, feature)
    vs = self.WV(embs) # (seq_len, feature)
    scores = t.mm(qs, ks.transpose(0, 1)) # (seq_len, seq_len)
    scores = t.softmax(scores / self.squre_dk, 1) # (seq_len, seq_len)
    result = t.mm(scores, vs) # (seq_len, feature)
    return result, scores.detach()

class Model_Self_Att(Model_TF_Decoder):
  def init_layers(self):
    self.selfatt_layer = Self_Att(self.feature)

  # decoder_inputs: (n, 1, feature)
  # return: (1, feature)
  def seq_output(self, decoder_inputs):
    decoder_inputs = t.stack(decoder_inputs) # decoder_inputs: (n, 1, feature)
    seq_len, batch, feature = decoder_inputs.shape
    decoder_inputs = decoder_inputs.view(seq_len, feature)
    pos_codings = position_encoding(decoder_inputs) # (n, feature)
    decoder_inputs = decoder_inputs + pos_codings # (n, feature)
    # decoder_inputs = decoder_inputs.view(seq_len, 1, feature) # (n, 1, feature)
    out, _ = self.selfatt_layer(decoder_inputs.float()) # (n, feature) 
    out = out[-1] # (feature)
    return out.view(1, self.feature)

