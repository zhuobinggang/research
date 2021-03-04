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
