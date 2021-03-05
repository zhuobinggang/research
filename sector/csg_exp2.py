from csg_exp import *

# =========== 倒序二次训练

class Dataset_Double_Order_Matters(Dataset_Single_Sentence_True):
  def inpt_from_sentences_pair(self, ss1, ss2):
    l = self.joined(ss1) # string
    r = self.joined(ss2)
    l_token_ids = self.token_encode(l)
    r_token_ids = self.token_encode(r)
    return [self.cls_id] + l_token_ids + [self.sep_id] + r_token_ids

  def __getitem__(self, idx):
    (l,r), label = self.__getitem_org__(idx)
    inpt = self.inpt_from_sentences_pair(l, r)
    inpt_revs = self.inpt_from_sentences_pair(r, l)
    return (inpt ,label), (inpt_revs ,label)

class Train_DS_Double_Order_Matters(Dataset_Double_Order_Matters):
  def init_datas_hook(self):
    self.datas = data.read_trains()

class Loader_Double_Order_Matters(CSG.Loader):
  # return: left: (batch, 128, 300), right: (batch, 128, 300), label: (batch)
  def __next__(self):
    if self.start == len(self.ds):
      self.start = 0
      raise StopIteration()
    results = []
    end = min(self.start + self.batch_size, len(self.ds))
    for i in range(self.start, end):
      r, r_revs = self.ds[i]
      results += [r, r_revs]
    self.start = end

    idss = [ids for ids, label in results]
    labels = t.LongTensor([label for ids, label in results])
    idss_paded, attend_mark = self.ds.pad_with_attend_mark(idss)

    return (t.LongTensor(idss_paded), t.LongTensor(attend_mark)), labels

def get_datas_order_matters(test):
  # Get Datas for Single_Sentence_CSG
  ld = Loader_Double_Order_Matters(Train_DS_Double_Order_Matters(), 2)
  testld = Loader_Single_Sentence_True(Test_DS_Single_Sentence_True(), 4)
  devld = Loader_Single_Sentence_True(Dev_DS_Single_Sentence_True(), 4)
  if test:
    ld.dataset.datas = ld.dataset.datas[:30]
    testld.dataset.datas = testld.dataset.datas[:15]
    devld.dataset.datas = devld.dataset.datas[:15]
  mess = []
  for i in range(2 if test else 5):
    m = CSG.Model()
    loss = runner.train_simple(m, ld, 1)
    runner.logout_info(f'Trained order matter model_{i} only one epoch, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result(m, testld)
    devdic = runner.get_test_result(m, devld)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['order_matters_mess'] = mess
