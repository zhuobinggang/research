import random

def generate_data(num):
  input_range = [1, 19]
  seq_size_range = [3, 5]
  result = []
  for i in range(num):
    temp = []
    for j in range(random.randint(seq_size_range[0], seq_size_range[1])):
      temp.append(random.randint(input_range[0], input_range[1]))
    result.append(temp)
  return result

def regenerate_train_and_test(train_lines = 900, test_lines = 100):
  train_data = generate_data(train_lines)
  test_data = generate_data(test_lines)
  with open ('train.txt', 'w') as f: 
    for data in train_data:
      f.write(str(data).replace('[','').replace(']','') + '\n')
  with open ('test.txt', 'w') as f: 
    for data in test_data:
      f.write(str(data).replace('[','').replace(']','') + '\n')

def read_data(filename='train.txt'):
  result = []
  with open (filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
       result.append(list(map(lambda x: int(x), line.rstrip().split(','))))
  return result
