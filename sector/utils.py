
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


