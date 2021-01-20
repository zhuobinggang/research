import seaborn as sns # for data visualization
import matplotlib.pyplot as plt
import torch as t

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
