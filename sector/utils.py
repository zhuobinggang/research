import seaborn as sns # for data visualization
import matplotlib.pyplot as plt
import torch as t
import time

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


def segment_by_score_matrix(mat):
  if len(mat) < 2:
    print(f'Should not segment a list with length {len(mat)}')
    return []
  if not hasattr(mat, 'shape'): 
    mat = t.tensor(mat)
  n = mat.shape[0]
  avg_score = (mat.triu(1).sum() / (n * (n - 1) / 2 )).item()
  results = []
  start = 0
  for i in range(1,n):
    score_against_prev_cluster  = 0
    for j in range(start, i): 
      score_against_prev_cluster += mat[j,i].item()
    avg_score_temp = score_against_prev_cluster / (i - start)
    if avg_score_temp < avg_score:
      start = i
      results.append(start)
  return results


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
    

def get_result_ids_by_datas(datas, m):
  return [segment_by_score_matrix(m.dry_run(ss)) for (ss, _, _) in datas]

def get_mats_by_datas(datas, m):
  start = time.time()
  results = [m.dry_run(ss) for (ss, _, _) in datas]
  end = time.time()
  print(f'Got outputs by datas and model, Time cost: {end - start} seconds')
  return results

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


def cal_Pk_by_model(datas, m):
  outs = get_result_ids_by_datas(datas, m)
  return cal_Pk(datas, outs)
