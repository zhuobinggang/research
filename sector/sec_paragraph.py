from sector_split import *

class Sec_Para(Sector_Split):
  def get_loss(self, mass): 
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    loss = []
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      # ss: [s,s,s,s]
      # ls: [0,1,0,0]
      # pos: number start from 0
      cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
      # NOTE: 去头去尾，必要操作，因为对应是错位的
      seps = seps[:-1] # 最后一个SEP不需要
      ls = ls[1:] # 第一个label不需要
      # Convert ls to Tensor
      ls = t.LongTensor(ls) # (ss_len), (0 or 1)
      if GPU_OK:
        ls = ls.cuda()
      assert ls.shape[0] == seps.shape[0]
      # 稍微做一点tricky的事情，将其他loss(除了中间那个) * 0.5
      o = self.classifier(seps) #(ss_len, 1)
      loss_per_sentence = [self.cal_loss(o_item.view(1, 1), l_item.view(1)) for (o_item, l_item) in zip(o, ls)]
    return loss
