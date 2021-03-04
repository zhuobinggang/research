from csg_bert import *

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

def run(m, epoch):
  # train
  losses = runner.train_simple(m, ld, epoch)
  testdic = runner.get_test_result(m, testld)
  devdic = runner.get_test_result(m, devld)
  return losses, testdic, devdic

def get_csg_128_normal_results():
  org_res = []
  for i in range(8):
    m = Model()
    loss, testdic, devdic = run(m, 2)
    org_res.append((testdic, devdic))
  return org_res

def analyse(org_res):
  test_fs = []
  test_baccs = []
  dev_fs = []
  dev_baccs = []
  for testdic, devdic in org_res:
    test_fs.append(testdic['f1'])
    dev_fs.append(devdic['f1'])
    test_baccs.append(testdic['bacc'])
    dev_baccs.append(devdic['bacc'])
  return test_fs, test_baccs, dev_fs, dev_baccs
    
def verbose_check(m):
  m.verbose = True
  dic = runner.get_test_result(m, devld)
  m.verbose = False
  return dic

