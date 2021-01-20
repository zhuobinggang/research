import random
import pickle

path = 'data.txt'

def generate_and_save(rows = 50):
  save(generate_rows(rows))


def generate_and_save_sorted(rows = 50):
  save(generate_rows(rows, True))

def generate_rows(row_num = 50, sort = False):
  result = []
  for i in range(0, row_num):
    result.append(generate_datas(sort))
  return result 

def generate_datas(sort = False):
  results = [ ]
  labels = [ ]
  start = 0
  cluster_index = 0
  for i in range(0, random.randint(3, 7)): # how many clusters
    cluster_index += 1
    for j in range(0, random.randint(2, 5)): # how many numbers in one cluster
      start += random.randint(1,3) # mini step
      results.append((cluster_index, start))
    start += random.randint(5,7)
  if not sort:
    random.shuffle(results)
  datas = [num for (_, num) in results]
  labels = attend_map(results)
  return datas, labels
      
def attend_map(datas):
  result = [ ]
  temp = [0] * len(datas)
  for _ in range(0, len(datas)):
    result.append(temp.copy())
  for i, (query_id, _) in enumerate(datas):
    for j, (target_id, _) in enumerate(datas):
      if target_id == query_id:
        result[i][j] = 1
  return result
  

def beutiful_print(mat):
  for row in mat:
    should_print = []
    for item in row: 
      should_print.append(item)
    print(should_print)

def save(a):
  with open(path, "wb") as fp:   #Pickling
    pickle.dump(a, fp)

def read():
  with open(path, "rb") as fp:   # Unpickling
    return  pickle.load(fp)

def run_example():
  generate_and_save(50)
  datas = read_data()
  for nums, labels in datas:
    pass
    
