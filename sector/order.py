from exp5 import *

class Ordering_Only(Double_Sentence_Plus_Ordering):
  def train(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      if pos != 1:
        print(f'Warning: pos={pos}, Will not train for {ss}')
      else:
        # Ordering
        ss_disturbed = ss.copy()
        if random.randrange(100) > 50: # 1/2的概率倒序
          random.shuffle(ss_disturbed)
        else:
          pass
        ordering_embs.append(self.pool_policy(ss_disturbed, pos))
        if ss_disturbed == ss:
          ordering_labels.append(0)
        else:
          ordering_labels.append(1)
    ordering_embs = t.stack(ordering_embs)
    ordering_labels = t.LongTensor(ordering_labels)
    if GPU_OK:
      ordering_labels = ordering_labels.cuda()
    o_ordering = self.classifier2(ordering_embs) # (batch, 1)
    loss = self.cal_loss(o_ordering, ordering_labels)
    self.zero_grad()
    loss.backward()
    self.optim.step()
    self.print_train_info(o, labels, loss.detach().item())
    return loss.detach().item()

  @t.no_grad()
  def dry_run(self, mass):
    batch = len(mass)
    sss, labels, poss = handle_mass(mass) 
    ordering_embs = []
    ordering_labels = []
    for ss, pos in zip(sss, poss):
      if pos != 1:
        print(f'Warning: pos={pos}, Will not train for {ss}')
      else:
        # Ordering
        ss_disturbed = ss.copy()
        if random.randrange(100) > 50: # 1/2的概率倒序
          random.shuffle(ss_disturbed)
        else:
          pass
        ordering_embs.append(self.pool_policy(ss_disturbed, pos))
        if ss_disturbed == ss:
          ordering_labels.append(0)
        else:
          ordering_labels.append(1)
    ordering_embs = t.stack(ordering_embs)
    ordering_labels = t.LongTensor(ordering_labels)
    if GPU_OK:
      ordering_labels = ordering_labels.cuda()
    o = self.classifier2(ordering_embs) # (batch, 1)
    self.print_train_info(o, ordering_labels, -1)
    return fit_sigmoided_to_label(o), ordering_labels

def run():
  init_G_Symmetry(1, sgd = True) 
  G['m'] = m = Ordering_Only(rate=0)
  get_datas(0, 1, f'1:1 Ordering Only, flrate={m.fl_rate}')
  get_datas(1, 1, f'1:1 Ordering Only, flrate={m.fl_rate}')
