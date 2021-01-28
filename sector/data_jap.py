def read_trains():
  filenames = ['sansirou.txt', 'sorekara.txt', 'mon.txt', 'higan.txt', 'gyoujin.txt']
  paths = [f'datasets/{name}' for name in filenames]
  for path in paths:
    with open(path) as f:
      f.readlines()
