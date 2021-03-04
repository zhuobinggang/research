import csg_bert as M 

def run(times):
  results = []
  for _ in range(0, times):
    m = M.Model()
    _, rs, ls = M.run(m)
    results.append({'result': rs, 'loss': ls})
  return results
