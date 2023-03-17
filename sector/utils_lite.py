import numpy as np
import torch
import requests
import matplotlib.pyplot as plt
import seaborn as sns # for data visualization
# plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams.update({
  # 'font.size': 5,
  'font.family': 'Noto Sans CJK JP'
})
t = torch
GPU_OK = t.cuda.is_available()

def position_encoding_ddd(t, i, d):
  k = int(i/2)
  omiga = 1 / np.power(10000, 2 * k / d)
  even = (i / 2).is_integer()
  return np.sin(omiga * t) if even else np.cos(omiga * t)

# seq: (seq_len, feature)
# return: (seq_len, feature)
def position_encoding(seq):
  embs = []
  for t, data in enumerate(seq):
    d = data.shape[0]
    pos_emb = [position_encoding_ddd(t, i, d) for i in range(0, d)]
    pos_emb = torch.tensor(pos_emb)
    embs.append(pos_emb)
  embs = torch.stack(embs)
  return embs.cuda() if GPU_OK else embs

def position_matrix(seq_len, feature):
  d = feature
  embs = []
  for j in range(seq_len):
    pos_emb = [position_encoding_ddd(j, i, d) for i in range(d)]
    pos_emb = torch.tensor(pos_emb)
    embs.append(pos_emb)
  embs = torch.stack(embs)
  return embs.cuda() if GPU_OK else embs

# def request_my_logger(dic, desc = 'No describe'):
#   try:
#     url = "https://hookb.in/b9xlr2GnnjC3DDogQ0jY"
#     dic['desc'] = desc
#     requests.post(url, json=dic)
#   except:
#     print('Something went wrong in request_my_logger()')


# head_scores: (batch, head, seq_len, seq_len)
# info: (seq_len)
def draw_head_attention(head_scores, info, cls_pos = 0, path='dd.png', desc = ''):
  mat = []
  head_scores = head_scores[0]
  head, seq_len, _ = head_scores.shape
  avg_scores = head_scores.sum(dim=0) / head # (seq_len, seq_len)
  avg_scores = avg_scores[cls_pos] # (seq_len)
  xs = info
  ys = []
  for i in range(head):
    ys.append(f'head_{i}')
    score = head_scores[i] # (seq_len, seq_len)
    score = score[cls_pos] # (seq_len)
    mat.append(score) 
  mat.append(avg_scores)
  ys += ['avg']
  mat = [row.tolist() for row in mat]
  for row in mat:
    row.pop(cls_pos)
  output_heatmap(np.transpose(mat), ys, xs, path, desc)

def output_heatmap(mat, xs, ys, path = 'dd.png', desc = ''):
  if len(ys) > 16:
    print(f'Warning: too long sequence: {len(ys)}')
    # sns.set(font_scale=0.5)
  else:
    # 5sns.set(font_scale=1.0)
    pass
  plt.clf()
  sns.heatmap(mat, xticklabels=xs, yticklabels=ys)
  plt.text(0 , -0.5, desc)
  plt.savefig(path)


def fit_sigmoided_to_label(out):
  assert len(out.shape) == 2
  results = []
  for item in out:
    assert item >= 0 and item <= 1
    if item < 0.5:
      results.append(0) 
    else:
      results.append(1) 
  return t.LongTensor(results)


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
