import model
import data_operator as data
import torch as t
from contextlib import redirect_stdout

hidden_state = 256
input_state = 256
epoch = 20
path = f'save/model_256.tch' 
trains = data.read_data()
print(f'Run Script Log: hidden_state = {hidden_state}, input_state = {input_state}, epoch = {epoch}, data rows = {len(trains)}, save path = {path}')
m = model.Model_V2(hidden_state, input_state)
m.SGD_train(trains, epoch)
t.save(m, path)


def strategy_train(ss, correct_indexs):
  length = len(ss)
  reversed_ids = list(reversed([(length - index) for index in correct_indexs]))
  return list(reversed(ss)), reversed_ids
