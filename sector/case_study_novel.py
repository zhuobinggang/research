from exp_novel import *
# 从朱庇特notebook启动
# from bertviz import head_view

novel_test = read_ld_test_from_chapters()

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
@t.no_grad()
def view_att(m, ss, stand = False, need_p = False):
  # ss, labels = row
  toker = m.toker
  bert = m.bert
  if not stand:
    combined_ids, sep_idxs = encode(ss, toker, [True, True, True, False])
    pool_idx = sep_idxs[1]
  else:
    combined_ids, sep_idxs = encode(ss, toker, [False, True, False, False])
    pool_idx = 0
  out = bert(combined_ids.unsqueeze(0).cuda(), output_attentions = True)
  attention = out.attentions 
  # tuple of (layers = 12, batch = 1, heads = 12, seq_len = n, seq_len = n)
  need = t.stack(attention) # (12, 1, 12, n, n)
  need = need.mean(0) # (1, 12, n, n)
  need = need[0] # (12, n, n)
  need = need.mean(0) # (n, n)
  need = need[pool_idx] # (n)
  atts = need.tolist()
  tokens = toker.convert_ids_to_tokens(combined_ids)
  # 同时输出p
  if need_p:
    last_hidden_state = out.last_hidden_state
    pooled_vector = last_hidden_state[0, pool_idx] # (768)
    p = m.classifier(pooled_vector).item() # p
    return p, tokens, atts
  else:
    return tokens, atts


def filter_special_tokens(wc):
    return [word for word in wc if word not in ['[SEP]','[CLS]']]

def rank(m, test_ds, stand = False):
    wordcloud = []
    for ss, labels in test_ds:
        tokens, atts = view_att(m, ss, stand)
        dd = list(reversed(sorted(zip(tokens, atts), key = lambda x: x[1])))
        best10 = [token for token, att in dd[:10]]
        wordcloud += best10
    return wordcloud

def filter_tokens_and_pick10(ps_tokens_atts, split = True):
    wc = []
    for p, tokens, atts in ps_tokens_atts:
        if (split and p > 0.5) or (not split and p < 0.5):
            dd = list(reversed(sorted(zip(tokens, atts), key = lambda x: x[1])))
            best10 = [token for token, att in dd[:10]]
            wc += best10
    return wc

def rank_by_p(m, test_ds, stand = False):
    ps_tokens_atts = []
    for ss, labels in test_ds:
        ps_tokens_atts.append(view_att(m, ss, stand, need_p = True))
    wc1 = filter_special_tokens(filter_tokens_and_pick10(ps_tokens_atts, split = True))
    wc0 = filter_special_tokens(filter_tokens_and_pick10(ps_tokens_atts, split = False))
    return wc1, wc0

# script
def generate_word_cloud():       
    from wordcloud import WordCloud
    dd = [word for word in aux_words if word not in ['[SEP]','[CLS]']]
    wordcloud = WordCloud(font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', collocations = False)
    wordcloud.generate(' '.join(dd))
    wordcloud.to_file('./aux.jpg')

def run():
    seed = 97
    m_aux = load_model(f'SEED_19_AUX01FL50E2_6')
    m_fl = load_model(f'SEED_19_FL50E2_6')
    wc1_aux, wc0_aux = rank_by_p(m_aux, novel_test, stand = False)
    wc1_fl, wc0_fl = rank_by_p(m_fl, novel_test, stand = True)
    return wc1_aux, wc0_aux, wc1_fl, wc0_fl
