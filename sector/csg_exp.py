# 实验场
import csg_bert as CSG
Loader = CSG.Loader
import data_jap_reader as data
import danraku_runner_simple as runner
from importlib import reload
import numpy as np
import torch as t
from itertools import chain
nn = t.nn

# ========= Single Sentence CSG

class Dataset_Single_Sentence_CSG(CSG.Dataset):
  def __getitem__(self, idx):
    (l,r), label = self.__getitem_org__(idx)
    l = self.joined(l) # string
    r = self.joined(r)
    l_token_ids = self.token_encode(l)
    r_token_ids = self.token_encode(r)
    # trim 
    l_token_ids, r_token_ids = self.trim_ids(l_token_ids, r_token_ids)
    # Pad and create attend mark and add special token
    results_ids, attend_mark = self.pad_and_create_attend_mark_with_special_token(l_token_ids, r_token_ids)
    return (results_ids, attend_mark), label

class Train_DS_Single_Sentence_CSG(Dataset_Single_Sentence_CSG):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS_Single_Sentence_CSG(Dataset_Single_Sentence_CSG):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS_Single_Sentence_CSG(Dataset_Single_Sentence_CSG):
  def init_datas_hook(self):
    self.datas = data.read_devs()

# ============= Single Sentence Cat Sentences

class Dataset_Single_Sentence(CSG.Dataset):
  def __getitem__(self, idx):
    (l,r), label = self.__getitem_org__(idx)
    l = self.joined(l) # string
    r = self.joined(r)
    l_token_ids = self.token_encode(l)
    r_token_ids = self.token_encode(r)
    return (l_token_ids, r_token_ids), label

  # ids: (batch * 2, ?)
  # return: (batch * 2, max_length)
  def pad_and_add_special_token(self, idss):
    max_len = max([len(ids) for ids in idss])
    results_idss = []
    attend_markss = []
    for ids in idss:
      topad = np.repeat(0, max_len - len(ids)).tolist()
      attend_marks = [1] + np.repeat(1, len(ids)).tolist() + topad
      results_ids = [self.cls_id] + ids + topad
      results_idss.append(results_ids)
      attend_markss.append(attend_marks)
    return results_idss, attend_markss

class Train_DS_Single_Sentence(Dataset_Single_Sentence):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS_Single_Sentence(Dataset_Single_Sentence):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS_Single_Sentence(Dataset_Single_Sentence):
  def init_datas_hook(self):
    self.datas = data.read_devs()

class Loader_Single_Sentence(CSG.Loader):
  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start == len(self.ds):
      self.start = 0
      raise StopIteration()
    results = []
    end = min(self.start + self.batch_size, len(self.ds))
    for i in range(self.start, end):
      results.append(self.ds[i])
    self.start = end

    lefts = [d[0][0] for d in results]
    rights = [d[0][1] for d in results]
    result_ids, attend_marks = self.ds.pad_and_add_special_token(lefts + rights)

    labels = t.LongTensor([d[1] for d in results])
    
    return (t.LongTensor(result_ids), t.LongTensor(attend_marks)), labels

class Train_DS_Single_Sentence(Dataset_Single_Sentence):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS_Single_Sentence(Dataset_Single_Sentence):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS_Single_Sentence(Dataset_Single_Sentence):
  def init_datas_hook(self):
    self.datas = data.read_devs()

class Model_Single_Sentence(CSG.Model):
  def get_should_update(self):
    return chain(self.bert.parameters(), self.classifier.parameters(), self.compresor.parameters())

  def init_hook(self):
    self.compresor = nn.Sequential(
      nn.Linear(self.bert_size * 2, self.bert_size),
      nn.LeakyReLU(0.1),
    )
  
  def processed_embs(self, embs):
    lefts, rights = embs.split(int(embs.shape[0] / 2)) # (batch, 768)
    combined = t.stack([t.cat((l,r)) for l,r in zip(lefts, rights)]) # (batch, 768 * 2)
    return self.compresor(combined) # (batch, 768)

# ================== Single Sentence


class Dataset_Single_Sentence_True(Dataset_Single_Sentence):
  def __getitem__(self, idx):
    (l,r), label = self.__getitem_org__(idx)
    l = self.joined(l) # string
    r = self.joined(r)
    l_token_ids = self.token_encode(l)
    r_token_ids = self.token_encode(r)
    return ([self.cls_id] + l_token_ids + [self.sep_id] + r_token_ids), label

  def pad_with_attend_mark(self, idss):
    max_len = max([len(ids) for ids in idss])
    idss_paded = []
    attend_mark = []
    for ids in idss:
      zeros = np.repeat(0, max_len - len(ids)).tolist()
      idss_paded.append(ids + zeros)
      ones = np.repeat(1, len(ids)).tolist()
      attend_mark.append(ones + zeros)
    return idss_paded, attend_mark

class Loader_Single_Sentence_True(CSG.Loader):
  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start == len(self.ds):
      self.start = 0
      raise StopIteration()
    results = []
    end = min(self.start + self.batch_size, len(self.ds))
    for i in range(self.start, end):
      results.append(self.ds[i])
    self.start = end

    idss = [ids for ids, label in results]
    labels = t.LongTensor([label for ids, label in results])
    idss_paded, attend_mark = self.ds.pad_with_attend_mark(idss)

    return (t.LongTensor(idss_paded), t.LongTensor(attend_mark)), labels

class Train_DS_Single_Sentence_True(Dataset_Single_Sentence_True):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Test_DS_Single_Sentence_True(Dataset_Single_Sentence_True):
  def init_datas_hook(self):
    self.datas = data.read_tests()

class Dev_DS_Single_Sentence_True(Dataset_Single_Sentence_True):
  def init_datas_hook(self):
    self.datas = data.read_devs()


# ============= Runner

G = {}

def get_datas_for_Single_Sentence_CSG(test):
  # Get Datas for Single_Sentence_CSG
  ld = Loader(Train_DS_Single_Sentence_CSG(), 4)
  testld = Loader(Test_DS_Single_Sentence_CSG(), 4)
  devld = Loader(Dev_DS_Single_Sentence_CSG(), 4)
  if test:
    ld.dataset.datas = ld.dataset.datas[:30]
    testld.dataset.datas = testld.dataset.datas[:15]
    devld.dataset.datas = devld.dataset.datas[:15]
  mess = []
  for i in range(2 if test else 5):
    m = CSG.Model()
    loss = runner.train_simple(m, ld, 2)
    runner.logout_info(f'Trained Single_Sentence_CSG model_{i}, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result(m, testld)
    devdic = runner.get_test_result(m, devld)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['single_sentence_csg_mess'] = mess

def get_datas_for_Single_Sentence_Cat(test):
  # Get Datas for Single_Sentence_CSG
  ld = Loader_Single_Sentence(Train_DS_Single_Sentence(), 4)
  testld = Loader_Single_Sentence(Test_DS_Single_Sentence(), 4)
  devld = Loader_Single_Sentence(Dev_DS_Single_Sentence(), 4)
  if test:
    ld.dataset.datas = ld.dataset.datas[:30]
    testld.dataset.datas = testld.dataset.datas[:15]
    devld.dataset.datas = devld.dataset.datas[:15]
  mess = []
  for i in range(2 if test else 5):
    m = Model_Single_Sentence()
    loss = runner.train_simple(m, ld, 2)
    runner.logout_info(f'Trained Single_Sentence model_{i}, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result(m, testld)
    devdic = runner.get_test_result(m, devld)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['single_sentence_mess'] = mess

def get_datas_for_Single_Sentence(test):
  # Get Datas for Single_Sentence_CSG
  ld = Loader_Single_Sentence_True(Train_DS_Single_Sentence_True(), 4)
  testld = Loader_Single_Sentence_True(Test_DS_Single_Sentence_True(), 4)
  devld = Loader_Single_Sentence_True(Dev_DS_Single_Sentence_True(), 4)
  if test:
    ld.dataset.datas = ld.dataset.datas[:30]
    testld.dataset.datas = testld.dataset.datas[:15]
    devld.dataset.datas = devld.dataset.datas[:15]
  mess = []
  for i in range(2 if test else 5):
    m = CSG.Model()
    loss = runner.train_simple(m, ld, 2)
    runner.logout_info(f'Trained Single_Sentence True model_{i}, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result(m, testld)
    devdic = runner.get_test_result(m, devld)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['single_sentence_true_mess'] = mess


def run(test = False):
  get_datas_for_Single_Sentence(test)

