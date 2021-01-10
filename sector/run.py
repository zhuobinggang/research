import model
import data_operator as data
import torch as t

trains = data.read_data()
m = model.Model_V2(256, 256)
# m.SGD_train(trains, 20)
m.SGD_train(trains[0:1], 2)
path = f'save/model_256.tch' 
t.save(m, path)
