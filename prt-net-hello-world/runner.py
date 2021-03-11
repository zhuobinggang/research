import data
D = data
import model_new as M

def train_ld():
  ds = data.Dataset()
  ds.datas = data.read_train()
  ld = data.Loader(ds, 1)
  return ld

def test_ld():
  ds = data.Dataset()
  ds.datas = data.read_test()
  ld = data.Loader(ds, 1)
  return ld


def gru_baseline_model():
  return M.Model()


def run(m, ld, epoch = 10):
  for i in range(epoch):
    loss = 0
    for inpts, labels in ld:
      loss += m.train(inpts, labels)
    print(f'epoch{i}, loss:{loss}')

def test(m):
  ld = test_ld()
  false_times = 0
  true_times = 0
  for inpts, labels in ld:
    outputs, targets = m.dry_run(inpts, labels)
    for out, tar in zip(outputs, targets):
      if out != tar:
        false_times += 1
      else:
        true_times += 1
  print(f'Tested, acc = {true_times / (true_times + false_times)}')
  return true_times, false_times

def script():
  ld = train_ld()
  m = gru_baseline_model()
  m.verbose = True
  run(m, ld ,1)
  return m
