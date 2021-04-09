from exp5 import *
import mainichi

class Ordering_Only(Double_Sentence_Plus_Ordering):

  def print_train_info(self, o, ordering_labels, loss):
    if self.verbose:
      o = [round(item, 2) for item in o.view(-1).tolist()]
      ordering_labels = ordering_labels.view(-1).tolist()
      print(f'Got: {o}, Want: {ordering_labels}')

  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      # if len(ss) != 4:
      #   print(f'Warning: pos={pos}, ss={ss[0]}')
      # else:
      #   pass
      # Ordering
      ss_disturbed = ss.copy()
      if random.randrange(100) > 50: # 1/2的概率倒序
        random.shuffle(ss_disturbed)
      else:
        pass
      ordering_embs.append(self.pool_policy(ss_disturbed, len(ss)))
      if ss_disturbed == ss:
        ordering_labels.append(0)
      else:
        ordering_labels.append(1)
    ordering_embs = t.stack(ordering_embs)
    ordering_labels = t.LongTensor(ordering_labels)
    if GPU_OK:
      ordering_labels = ordering_labels.cuda()
    o_ordering = self.classifier2(ordering_embs) # (batch, 1)
    o = o_ordering
    loss = self.cal_loss(o_ordering, ordering_labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, ordering_labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      # if len(ss) != 4:
      #   print(f'Warning: pos={pos}, ss={ss[0]}')
      # else:
      #   pass
      # Ordering
      ss_disturbed = ss.copy()
      if random.randrange(100) > 50: # 1/2的概率倒序
        random.shuffle(ss_disturbed)
      else:
        pass
      ordering_embs.append(self.pool_policy(ss_disturbed, len(ss)))
      if ss_disturbed == ss:
        ordering_labels.append(0)
      else:
        ordering_labels.append(1)
    ordering_embs = t.stack(ordering_embs)
    ordering_labels = t.LongTensor(ordering_labels)
    if GPU_OK:
      ordering_labels = ordering_labels.cuda()
    o_ordering = self.classifier2(ordering_embs) # (batch, 1)
    o = o_ordering
    self.print_train_info(o, ordering_labels, -1)
    return fit_sigmoided_to_label(o), ordering_labels

class Ordering_Sector(Ordering_Only):
  def pool_policy_sectoring(self, ss, pos):
    return B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, pos) # (784)

  def pool_policy_ordering(self, ss, pos):
    return self.pool_policy_sectoring(ss, len(ss)) # NOTE: 无视pos

  def get_sector_output(self, sss, poss):
    sector_embs = []
    for ss, pos in zip(sss, poss):
      sector_embs.append(self.pool_policy_sectoring(ss, pos)) # 用于判断分割点的embs
    sector_embs = t.stack(sector_embs) # (batch, feature)
    o_sector = self.classifier(sector_embs) # (batch, 1)
    return o_sector
  
  def get_sector_loss(self, sss, poss, sector_labels, return_output = True):
    o_sector = self.get_sector_output(sss, poss) # (batch, 1)
    sector_labels = t.LongTensor(sector_labels) # (batch), (0 or 1)
    if GPU_OK:
      sector_labels = sector_labels.cuda()
    sector_loss = self.cal_loss(o_sector, sector_labels)
    if return_output:
      return sector_loss, o_sector
    else:
      return sector_loss

  def get_ordering_output_and_label(self, sss, poss):
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      # if len(ss) != 4:
      #   print(f'Warning: pos={pos}, ss={ss[0]}')
      # else:
      #   pass
      # Ordering
      ss_disturbed = ss.copy()
      if random.randrange(100) > 50: # 1/2的概率倒序
        random.shuffle(ss_disturbed)
      else:
        pass
      ordering_embs.append(self.pool_policy_ordering(ss_disturbed, pos))
      if ss_disturbed == ss:
        ordering_labels.append(0)
      else:
        ordering_labels.append(1)
    ordering_embs = t.stack(ordering_embs)
    ordering_labels = t.LongTensor(ordering_labels)
    if GPU_OK:
      ordering_labels = ordering_labels.cuda()
    o_ordering = self.classifier2(ordering_embs) # (batch, 1)
    return o_ordering, ordering_labels

  def get_ordering_loss(self, sss, poss):
    o_ordering, ordering_labels = self.get_ordering_output_and_label(sss, poss)
    ordering_loss = self.cal_loss(o_ordering, ordering_labels)
    return ordering_loss

  def train(self, mass):
    batch = len(mass)
    sss, sector_labels, poss = handle_mass(mass) 
    sector_loss, sector_output = self.get_sector_loss(sss, poss, sector_labels, return_output = True)
    loss = self.get_ordering_loss(sss, poss) + sector_loss
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(sector_output, sector_labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, sector_labels, poss = handle_mass(mass) 
    sector_output = self.get_sector_output(sss, poss)
    self.print_train_info(sector_output, sector_labels, -1)
    return fit_sigmoided_to_label(sector_output), t.LongTensor(sector_labels)


class Ordering_Sector_Save_Dry_Run(Ordering_Sector):
  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, sector_labels, poss = handle_mass(mass) 
    sector_output = self.get_sector_output(sss, poss)
    # 保存dry_run结果到自身
    o_ordering, ordering_labels = self.get_ordering_output_and_label(sss, poss)
    self.dry_run_output += fit_sigmoided_to_label(o_ordering).view(-1).tolist()
    self.dry_run_labels += ordering_labels.view(-1).tolist()
    self.print_train_info(sector_output, sector_labels, -1)
    return fit_sigmoided_to_label(sector_output), t.LongTensor(sector_labels)

class Sector_SEP_Order_CLS(Ordering_Sector):
  def pool_policy_sectoring(self, ss, pos):
    return B.compress_by_ss_pos_get_sep(self.bert, self.toker, ss, pos) # (784)

  def pool_policy_ordering(self, ss, pos):
    return B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, len(ss)) # (784)

class Sector_SEP_Order_CLS_Save_Ordering_Result(Ordering_Sector_Save_Dry_Run):
  def pool_policy_sectoring(self, ss, pos):
    return B.compress_by_ss_pos_get_sep(self.bert, self.toker, ss, pos) # (784)

  def pool_policy_ordering(self, ss, pos):
    return B.compress_by_ss_pos_get_cls(self.bert, self.toker, ss, len(ss)) # (784)

# ===========================================

def run():
  init_G_Symmetry(1, sgd = True) 
  G['m'] = m = Ordering_Only(rate=0)
  get_datas(0, 1, f'1:1 Ordering Only, flrate={m.fl_rate}')
  get_datas(1, 1, f'1:1 Ordering Only, flrate={m.fl_rate}')

def run2vs2():
  init_G_Symmetry(2, sgd = True)
  G['m'] = m = Ordering_Only(rate=0)
  get_datas(0, 1, f'2:2 Ordering Only, flrate={m.fl_rate}')
  get_datas(1, 1, f'2:2 Ordering Only, flrate={m.fl_rate}')

def run_order_sector():
  init_G_Symmetry(2, sgd = True, batch = 2)
  for i in range(6):
    G['m'] = m = Ordering_Sector(rate=0)
    get_datas(i, 2, f'2:2 Ordering+Sector, flrate={m.fl_rate}')
    G['m'] = m = Ordering_Sector(rate=3)
    get_datas(i + 10, 2, f'2:2 Ordering+Sector, flrate={m.fl_rate}')

def run_save_dryrun():
  init_G_Symmetry(2, sgd = True, batch = 2)
  for i in range(6):
    G['m'] = m = Sector_SEP_Order_CLS(rate=0)
    get_datas(i, 2, f'2:2 Sector_SEP_Order_CLS, flrate={m.fl_rate}')
    R.request_my_logger({
      'ordering_result': U.cal_prec_rec_f1_v2(m.dry_run_output, m.dry_run_labels)
    }, 'dd')
    G['m'] = m = Ordering_Sector_Save_Dry_Run(rate=0)
    get_datas(i + 10, 2, f'2:2 Ordering_Sector_Save_Dry_Run+Sector, flrate={m.fl_rate}')
    R.request_my_logger({
      'ordering_result': U.cal_prec_rec_f1_v2(m.dry_run_output, m.dry_run_labels)
    }, 'dd')


def run():
  for i in range(20): 
    # init_G_Symmetry(2, sgd = True, batch = 2)
    # G['m'] = m = Double_Sentence_CLS(rate=0) # 2 vs 2
    # get_datas(i + 20, 2, f'2vs2')
    # init_G_Symmetry(2, sgd = True, batch = 2)
    # G['m'] = m = Sector_SEP_Order_CLS(rate=0) # 2 vs 2, ordering
    # get_datas(i + 40, 2, f'2vs2, plus ordering')
    init_G_Symmetry(1, sgd = True, batch = 4)
    G['m'] = m = Double_Sentence_CLS(rate=0) # 1 vs 1
    get_datas(i, 2, f'1vs1')
    G['m'] = m = Sector_SEP_Order_CLS(rate=0) # 1 vs 1
    get_datas(i + 20, 2, f'1vs1 ordering')


def init_G_Symmetry_Mainichi(half = 1, batch = 4):
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_trains())
  G['ld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)
  ds = data.Dataset(ss_len = half * 2, datas = mainichi.read_tests())
  G['testld'] = data.Loader_Symmetry_SGD(ds = ds, half = half, batch = batch)

def run_mainichi():
  init_G_Symmetry_Mainichi(half = 1, batch = 4)
  G['m'] = m = Double_Sentence_CLS(rate=0) # 1 vs 1
  get_datas(0, 1, '1 vs 1, mainichi news', with_dev = False)
  get_datas(1, 1, '1 vs 1, mainichi news', with_dev = False)
  init_G_Symmetry_Mainichi(half = 2, batch = 2)
  G['m'] = m = Double_Sentence_CLS(rate=0) # 1 vs 1
  get_datas(2, 1, '2 vs 2, mainichi news', with_dev = False)
  get_datas(3, 1, '2 vs 2, mainichi news', with_dev = False)
  
