from sector_split import *
import mainichi_paragraph as custom_data

def init_G_Symmetry_Mainichi_With_Valid_Dev(half = 2, batch = 1):
  G['ld'] = custom_data.load_customized_loader(file_name = 'train', half = half, batch = batch)
  G['testld'] = custom_data.load_customized_loader(file_name = 'test', half = half, batch = batch)
  G['validld'] = custom_data.load_customized_loader(file_name = 'valid', half = half, batch = batch)
  G['devld'] = custom_data.load_customized_loader(file_name = 'dev', half = half, batch = batch)

def get_datas_early_stop_and_parameter_ajust(index, epochs, desc, dic_to_send = None, url = None):
  valid_losses = [] # For early stop
  train_losses = []
  tested = []
  for i in range(epochs):
    train_losses += train_simple(G['m'], G['ld'], 1)
    valid_loss = cal_valid_loss(G['m'], G['validld'])
    valid_losses.append(valid_loss)
    dic_to_analyse = get_test_result_dic(G['m'], G['testld'])
    dic_to_analyse['index'] = i # Save index info
    dic_to_analyse['valid_loss'] = valid_loss # Save index info
    tested.append(dic_to_analyse)
  test_result = tested[np.argmin(valid_losses)]
  # test_result['valid_losses'] = valid_losses
  G['mess_list'].append(test_result) # 将valid loss最小对应的dic放进mess_list
  dic = test_result
  if dic_to_send is not None:
    dic = {**dic, **dic_to_send}
  else:
    pass
  R.request_my_logger(dic, desc, url)

# ================================== Auxiliary Methods ====================================

class Sec_Para(Sector_Split):
  def get_loss(self, mass): 
    batch = len(mass)
    sss, labels, poss = self.handle_mass(mass) 
    losses = [] # 所有batch内的part loss都flatten到这里来
    # labels = B.flatten_num_lists(labels) # 这里要保留所有所有[sep]的label
    for ss, ls, pos in zip(sss, labels, poss): # Different Batch
      # ss: [s,s,s,s]
      # ls: [0,1,0,0]
      # pos: number start from 0
      if pos == 0:
        print('Skip "Start Row". For detail to check https://github.com/zhuobinggang/research/tree/master/sector')
      cls, seps = B.compress_by_ss_get_special_tokens(self.bert, self.toker, ss)
      # NOTE: 去头去尾，必要操作，因为对应是错位的
      seps = seps[:-1] # 最后一个SEP不需要
      ls = ls[1:] # 第一个label不需要
      pos = pos - 1 # POS也要调整，因为现在掐头去尾了，pos要-1 (比如ss[s]s的时候，2要变为1)，需要注意的是，pos=0的时候，会变为-1，在处理auxiliary loss的时候，要考虑到这一点
      # Convert ls to Tensor
      ls = t.LongTensor(ls) # (ss_len), (0 or 1)
      if GPU_OK:
        ls = ls.cuda()
      assert ls.shape[0] == seps.shape[0]
      # 稍微做一点tricky的事情，将其他loss(除了中间那个) * 0.5
      o = self.classifier(seps) #(ss_len, 1)
      loss_part = [self.cal_loss(o_item.view(1, 1), l_item.view(1)) for (o_item, l_item) in zip(o, ls)]
      # NOTE: Augment loss at 'pos'
      loss_part = [(los * self.auxiliary_loss_rate if (index != pos) else los) for index, los in enumerate(loss_part)]
      losses += loss_part
    return losses


