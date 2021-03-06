from csg_exp2 import * 

class Dataset_Around_Split_Point(Dataset_Single_Sentence_True):
  def init_hook(self):
    print('DDDDDDDDDD')
    self.datas = []
    self.datas_processed = []
    self.datas_trimed = []
    self.tokens = 128
    self.max_size = 128
    self.init_toker()

  def process_datas(self, datas):
    self.datas = datas
    self.datas_processed = []
    length = len(datas)
    start = self.half_window_size - 1
    end = length - (self.half_window_size + 1)
    for i in range(start, end):
      l_and_r, label = self.__getitem_org__(i)
      self.datas_processed.append((l_and_r, label))

  def safe_get(self, datas, idx):
    return None if idx < 0 or idx > (len(datas) - 1) else datas[idx]

  def set_datas(self, datas):
    self.process_datas(datas)
    self.datas_trimed = []
    # Trim Redundant
    for i, (l_and_r, label) in enumerate(self.datas_processed):
      if label == 1:
        self.datas_trimed += [data for data in [self.safe_get(self.datas_processed, i-1), (l_and_r, label), self.safe_get(self.datas_processed, i+1)] if data is not None]

  def __len__(self):
    return len(self.datas_trimed)
    
  def __getitem__(self, idx):
    data = self.safe_get(self.datas_trimed, idx)
    if data is not None:
      (l,r), label = data
      l = self.joined(l) # string
      r = self.joined(r)
      l = self.token_encode(l)
      r = self.token_encode(r)
      return ([self.cls_id] + l + [self.sep_id] + r), label
    else:
      return None

class Train_DS_Around_Split_Point(Dataset_Around_Split_Point):
  def init_datas_hook(self):
    self.datas = data.read_trains()

def get_datas_trimed_redundant(test):
  # Get Datas for Single_Sentence_CSG
  ld = Loader_Single_Sentence_True(Train_DS_Around_Split_Point(2), 4)
  testld = Loader_Single_Sentence_True(Test_DS_Single_Sentence_True(2), 4)
  devld = Loader_Single_Sentence_True(Dev_DS_Single_Sentence_True(2), 4)
  if test:
    ld.dataset.datas = ld.dataset.datas[:30]
    testld.dataset.datas = testld.dataset.datas[:15]
    devld.dataset.datas = devld.dataset.datas[:15]
  mess = []
  for i in range(2 if test else 5):
    m = CSG.Model()
    loss = runner.train_simple(m, ld, 2)
    runner.logout_info(f'Trained Double_Sentence True model_{i}, loss={loss}')
    runner.logout_info(f'Start test_{i}...')
    testdic = runner.get_test_result(m, testld)
    devdic = runner.get_test_result(m, devld)
    runner.logout_info(f'Over test_{i}:')
    runner.logout_info(f'testdic: {testdic}, devdic: {devdic}')
    mess.append((loss, testdic, devdic))
  G['datas_trimed_redundant_mess'] = mess

def run(test = False):
  get_datas_trimed_redundant(test)
