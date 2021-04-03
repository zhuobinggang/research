from exp5 import *

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
      if len(ss) != 4:
        print(f'Warning: pos={pos}, ss={ss[0]}')
      else:
        pass
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
      if len(ss) != 4:
        print(f'Warning: pos={pos}, ss={ss[0]}')
      else:
        pass
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
  def get_sector_loss(self, sss, poss, sector_labels, return_output = True):
    sector_embs = []
    for ss, pos in zip(sss, poss):
      sector_embs.append(self.pool_policy(ss, pos)) # 用于判断分割点的embs
    sector_embs = t.stack(sector_embs) # (batch, feature)
    sector_labels = t.LongTensor(sector_labels) # (batch), (0 or 1)
    if GPU_OK:
      sector_labels = sector_labels.cuda()
    o_sector = self.classifier(sector_embs) # (batch, 1)
    sector_loss = self.cal_loss(o_sector, sector_labels)
    if return_output:
      return sector_loss, o_sector
    else:
      return sector_loss

  def get_ordering_loss(self, sss, poss):
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      if len(ss) != 4:
        print(f'Warning: pos={pos}, ss={ss[0]}')
      else:
        pass
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
    sector_loss, sector_output = self.get_sector_loss(sss, poss, sector_labels, return_output = True)
    self.print_train_info(sector_output, sector_labels, -1)
    return fit_sigmoided_to_label(sector_output), sector_labels

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
  init_G_Symmetry(2, sgd = True)
  for i in range(5):
    G['m'] = m = Ordering_Sector(rate=0)
    get_datas(i, 2, f'2:2 Ordering+Sector, flrate={m.fl_rate}')
    G['m'] = m = Ordering_Sector(rate=3)
    get_datas(i, 2, f'2:2 Ordering+Sector, flrate={m.fl_rate}')



