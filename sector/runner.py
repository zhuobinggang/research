import utils as U
U.init_logger('textseg.log')
import textseg as model
m = model.Model(256, 256)
import dataset as ds
trainld = ds.Loader(ds.WikiSectionDataset(), 8)
testld = ds.Loader(ds.WikiSectionDataset(True), 1)

def run(big_epoch, step):
  results = U.train_by_data_loader_check(m, trainld, testld, big_epoch, step)
  return results

