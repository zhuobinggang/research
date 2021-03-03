import csg_bert as M
from transformers import BertModel, BertTokenizer
import data_jap_reader as data
from pyknp import Juman
from importlib import reload
import danraku_runner_simple as runner
Loader = M.Loader

class JumanTokenizer():
  def __init__(self):
    self.juman = Juman()
  
  def tokenize(self, text):
    result = self.juman.analysis(text)
    return ' '.join([mrph.midasi for mrph in result.mrph_list()])


class Dataset(M.Dataset):
  def token_encode(self, text):
    return self.toker.encode(self.juman.tokenize(text), add_special_tokens = False)

  def init_toker(self):
    toker = BertTokenizer("/home/taku/projects/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt", do_lower_case=False, do_basic_tokenize=False)
    self.toker = toker
    self.pad_id = toker.pad_token_id
    self.cls_id = toker.cls_token_id
    self.sep_id = toker.sep_token_id
    self.juman = JumanTokenizer()

class Train_DS(Dataset):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS(Dataset):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS(Dataset):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Model(M.Model):
  def init_bert(self):
    print('Init kuro bert')
    self.bert = BertModel.from_pretrained("/home/taku/projects/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers")
    self.bert.train()


# ============

batch_size = 4

ld = Loader(Train_DS(), batch_size)
testld = Loader(Test_DS(), batch_size)
devld = Loader(Dev_DS(), batch_size)

def set_test():
  ld.dataset.datas = ld.dataset.datas[:30]
  testld.dataset.datas = testld.dataset.datas[:15]
  devld.dataset.datas = devld.dataset.datas[:15]

# return: (m, (prec, rec, f1, bacc), losss)
def run_test(m, epoch = 2):
  set_test()
  m.verbose = True
  return runner.run(m, ld, testld, devld, epoch=epoch, batch=batch_size)

def run(m, epoch = 2):
  return runner.run(m, ld, testld, devld, epoch=epoch, batch=batch_size)

def run_at_night():
  rs = []
  ls = []
  for i in [1]:
    m = Model()
    _, results, losss = run(m)
    rs.append(results)
    ls.append(losss)
    t.save(m, f'save/csg_bert_{i}.tch')
  return rs, ls
