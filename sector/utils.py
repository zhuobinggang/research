import seaborn as sns # for data visualization
import matplotlib.pyplot as plt
import torch as t
import time
import logging

def init_logger(path):
  logging.basicConfig(
    filename=path,
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


def print_table(results, step):
  counter = 0
  acc_accuracy = 0
  acc_ex_rate = 0
  acc_short_rate = 0
  acc_repeat_rate = 0
  for accuracy, ex_rate, short_rate, repeat_rate in results:
    acc_accuracy += accuracy
    acc_ex_rate += ex_rate
    acc_short_rate += short_rate
    acc_repeat_rate += repeat_rate
    counter += step
    print(f'|{counter}|{round(accuracy, 2)}|{round(ex_rate, 2)}|{round(short_rate, 2)}|{round(repeat_rate, 2)}|')
  # Average
  length = len(results)
  print(f'|平均|{round(acc_accuracy / length, 2)}|{round(acc_ex_rate / length, 2)}|{round(acc_short_rate/ length, 2)}|{round(acc_repeat_rate / length, 2)}|')



def output_heatmap(mat, xs, ys, path = 'dd.png'):
  plt.clf()
  sns.heatmap(mat, xticklabels=xs, yticklabels=ys)
  plt.savefig(path)


def save(m, name='NOTHING'):
  path = f'save/model_{name}.tch' 
  t.save(m, path)


def dry_run_then_output(m, datas, index):
    mat = m.dry_run(datas[index][0])
    m.output(mat, datas[index][0], datas[index][1])


def beuty_print_sentences(ss, ids = []):
  results = [s[0: 80] for s in ss]
  for index in ids:
    results[index] = '$ ' + results[index]
  for s in results:
    print(f'{s}\n')

def beuty_print_data(datas, index):
  ss, ids, _ = datas[index]
  beuty_print_sentences(ss, ids)





def get_k_by_datas(datas):
  lengths_and_segment_counts = [(len(ss), segment_count)  for (ss, _ ,segment_count) in datas]
  total_lengths = sum([length for (length,_) in lengths_and_segment_counts])
  total_count = sum([count for (_,count) in lengths_and_segment_counts])
  return (total_lengths / total_count) / 2 # We set k to half of the average segment length

def try_alarm_miss_times(o_ids, t_ids, length, k):
  k = int(k)
  try_times = 0
  alarm_times = 0
  miss_times = 0
  for i in range(0, length - k):
    try_times += 1
    l_probe = i
    r_probe = l_probe + k
    refs = [item for item in t_ids if (item - 1) in range(l_probe, r_probe + 1)]
    hyps = [item for item in o_ids if (item - 1) in range(l_probe, r_probe + 1)]
    if len(refs) < len(hyps):
      alarm_times += 1
    if len(refs) > len(hyps):
      miss_times += 1
    else:
      pass # ture_times = try_times - (alarm_times + miss_times)
  return try_times, alarm_times, miss_times
    

# def get_result_ids_by_datas(datas, m):
#   return [segment_by_score_matrix(m.dry_run(ss)) for (ss, _, _) in datas]

def get_mats_by_loader(loader, m):
  loader.batch_size = 1
  loader.start = 0
  start = time.time()
  results = [m.dry_run(inpts) for (inpts, _) in loader]
  end = time.time()
  print(f'Got outputs by datas and model, Time cost: {end - start} seconds')
  return results


# def cal_Pk_by_ids(oids, tids, k):
#   all_try_times = 0
#   all_alarm_times = 0 
#   all_miss_times = 0
#   for index, (ss, t_ids, _) in enumerate(datas):
#     o_ids = outs[index]
#     try_times, alarm_times, miss_times = try_alarm_miss_times(o_ids, t_ids, len(ss), k)
#     all_try_times += try_times
#     all_alarm_times += alarm_times
#     all_miss_times += miss_times
#   print(f'Pk = {(all_alarm_times + all_miss_times) / all_try_times}')
#   return all_try_times, all_alarm_times, all_miss_times



# datas: [(ss, ids, segmentation_count)]
# outs: [[id]]
def cal_Pk(datas, outs):
  all_try_times = 0
  all_alarm_times = 0 
  all_miss_times = 0
  k = get_k_by_datas(datas)
  for index, (ss, t_ids, _) in enumerate(datas):
    o_ids = outs[index]
    try_times, alarm_times, miss_times = try_alarm_miss_times(o_ids, t_ids, len(ss), k)
    all_try_times += try_times
    all_alarm_times += alarm_times
    all_miss_times += miss_times
  print(f'Pk = {(all_alarm_times + all_miss_times) / all_try_times}')
  return all_try_times, all_alarm_times, all_miss_times

def get_batch_from_datas(datas, start_index, batch_size = 4):
  inpts = []
  labels = []
  for d in datas:
    inpts.append(d[0])
    labels.append(ids = d[1])
  return inpts, labels


def train_by_data_loader(m, loader, epoch = 5, logger = print):
  loader.start = 0
  length = len(loader.ds)
  start = time.time()
  for e in range(epoch):
    logger(f'start epoch{e}')
    loader.shuffle()
    counter = 0
    for inpts, labels in loader:
      o,l = m.train(inpts, labels)
      counter += loader.batch_size
      logger(f'{counter}/{length}')
  end = time.time()
  logger(f'Trained! Epochs: {epoch}, Batch size: {loader.batch_size}, dataset length: {length}, Time cost: {end - start} seconds')


def train_by_data_loader_check(m, loader, testloader, big_epoch = 10, output_step = 2):
  results = []
  for _ in range(big_epoch):
    train_by_data_loader(m, loader, output_step, logging.debug)
    res = m.cal_Pk_by_loader(testloader)
    results.append(res)
    print(f'res = {res}')
  return results


# === danraku ===

def train_by_data_loader_danraku(m, loader, epoch = 5, logger = print):
  loss_per_epoch = []
  loader.start = 0
  length = len(loader.dataset)
  start = time.time()
  for e in range(epoch):
    logger(f'start epoch{e}')
    loader.shuffle()
    total_loss = 0
    counter = 0
    for inpts, labels in loader:
      counter += 1
      logger(f'{loader.start}/{length}')
      total_loss += m.train(inpts, labels)
    avg_loss = total_loss / counter
    loss_per_epoch.append(avg_loss)
  end = time.time()
  logger(f'Trained! Epochs: {epoch}, Batch size: {loader.batch_size}, dataset length: {length}, Time cost: {end - start} seconds')
  return loss_per_epoch

def train_by_data_loader_danraku_origin(m, loader, epoch = 5, logger = print):
  loss_per_epoch = []
  loader.start = 0
  length = len(loader.dataset)
  start = time.time()
  for e in range(epoch):
    logger(f'start epoch{e}')
    # loader.shuffle()
    total_loss = 0
    counter = 0
    for inpts, labels in loader:
      counter += loader.batch_size
      logger(f'{counter}/{length}')
      total_loss += m.train(inpts, labels)
    avg_loss = total_loss / len(loader)
    loss_per_epoch.append(total_loss)
  end = time.time()
  logger(f'Trained! Epochs: {epoch}, Batch size: {loader.batch_size}, dataset length: {length}, Time cost: {end - start} seconds')
  return loss_per_epoch


def get_avg_loss(m, ld):
  ld.start = 0
  ld.batch_size = 1
  loss = 0
  counter = 0
  for inpts,labels in ld:
    counter += 1
    loss += m.get_loss(inpts, labels).item()
  return loss / counter

def get_test_results_origin(m, testld):
  targets = []
  results = []
  for inpts, labels in testld:
    targets += labels
    results.append(m.dry_run(inpts))
  return results, targets


def get_test_results(m, testld):
  testld.start = 0
  testld.batch_size = 1
  targets = []
  results = []
  for inpts, labels in testld:
    targets += labels
    results.append(m.dry_run(inpts))
  return results, targets


def cal_prec_rec_f1(results, targets):
  true_positive = 0
  false_positive = 0
  trues = 0
  for guess, target in zip(results, targets):
    if target == guess:
      trues += 1
    if target == 1:
      if guess == 1:
        true_positive += 1
      else:
        false_positive += 1
  prec = trues / len(results)
  rec = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  return prec, rec, f1

def cal_prec_rec_f1_v2(results, targets):
  TP = 0
  FP = 0
  FN = 0
  TN = 0
  for guess, target in zip(results, targets):
    if guess == 1:
      if target == 1:
        TP += 1
      elif target == 0:
        FP += 1
    elif guess == 0:
      if target == 1:
        FN += 1
      elif target == 0:
        TN += 1
  prec = TP / (TP + FP) if (TP + FP) > 0 else 0
  rec = TP / (TP + FN) if (TP + FN) > 0 else 0
  f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
  balanced_acc_factor1 = TP / (TP + FN) if (TP + FN) > 0 else 0
  balanced_acc_factor2 = TN / (FP + TN) if (FP + TN) > 0 else 0
  balanced_acc = (balanced_acc_factor1 + balanced_acc_factor2) / 2
  return prec, rec, f1, balanced_acc

