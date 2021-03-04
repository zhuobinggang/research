import csg_bert_runner as M 
runner = M.runner

def check_epoch_stop():
  m = M.Model()
  losses = []
  devdics = []
  for i in range(5):
    losses.append(runner.train_simple(m, M.ld, 1)[0])
    devdics.append(runner.get_test_result(m, M.devld))
  return losses, devdics
