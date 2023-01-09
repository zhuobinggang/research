from exp_novel import *
# 从朱庇特notebook启动
# from bertviz import head_view

novel_test = read_ld_test_from_chapters()

# SEED = 19
# SEED = 666
def get_results(testset, seed, index):
    m = load_model(f'SEED_{seed}_AUX01FL50E2_{index}')
    y_true1, y_pred1 = test(testset, m)
    m = load_model(f'SEED_{seed}_FL50E2_{index}')
    y_true2, y_pred2 = test(testset, m)
    return y_true1, y_pred1, y_true2, y_pred2

def get_case1(labels, y1, y2, testset):
    res = []
    for label, auxfl, fl, case in zip(labels, y1, y2, testset):
        if label == 1 and auxfl > 0.5 and fl < 0.5:
            res.append((auxfl - 0.5 + 0.5 - fl, auxfl, fl, case))
    return res

def get_case2(labels, y1, y2, testset):
    res = []
    for label, auxfl, fl, case in zip(labels, y1, y2, testset):
        if label == 1 and auxfl < 0.5 and fl > 0.5:
            res.append((fl - 0.5 + 0.5 - auxfl, auxfl, fl, case))
    return res

# labels, y1, _, y2 = get_results(novel_test, 14)


### 可视化注意力 ###
def view_att(m, ss, stand = False):
  # ss, labels = row
  toker = m.toker
  bert = m.bert
  if not stand:
    combined_ids, sep_idxs = encode(ss, toker, [True, True, True, False])
    pool_idx = sep_idxs[1]
  else:
    combined_ids, sep_idxs = encode(ss, toker, [False, True, False, False])
    pool_idx = 0
  attention = bert(combined_ids.unsqueeze(0).cuda(), output_attentions = True).attentions 
  # tuple of (layers = 12, batch = 1, heads = 12, seq_len = n, seq_len = n)
  need = t.stack(attention) # (12, 1, 12, n, n)
  need = need.mean(0) # (1, 12, n, n)
  need = need[0] # (12, n, n)
  need = need.mean(0) # (n, n)
  need = need[pool_idx] # (n)
  atts = need.tolist()
  tokens = toker.convert_ids_to_tokens(combined_ids)
  return tokens, atts

# m = load_model(f'SEED_19_AUX01FL50E2_6')
# m = load_model(f'SEED_19_AUX01FL50E2_6')
def run():
    m = load_model(f'SEED_19_AUX01FL50E2_6')
    ss, labels = novel_test[1]
    view_att_my(m, ss)



