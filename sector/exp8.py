# 选出最受瞩目的token，用wordcloud来展示
from sec_paragraph import get_att_ours
from exp7 import load_mld
import torch as t
import numpy as np
font = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
# font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
from wordcloud import WordCloud

def load_m():
    return t.load('save/r01_fl50_0.tch')
    # return t.load('save/analyse_r02_para.tch')

# 从mld选出取出正确率最高的100个例子
def get_info_ranking_by_posibility():
    m = load_m()
    pcases = cases_positive(load_mld())
    need = cases_ranking_from_max(m, pcases)
    return need, m.toker

def cases_positive(ld):
    results = []
    for case in ld:
        ss, labels, pos = case[0]
        if labels[pos] == 1:
            results.append(case)
    return results

def cases_ranking_from_max(m, cases):
    need = []
    for case in cases:
        atts, idss, results, targets, labelss = get_att_ours(m, case)
        need.append((results[0], case, atts[0], idss[0]))
    need = list(reversed(sorted(need, key = lambda x : x[0])))
    return need

# 从所有case里面每个取出10个token，然后将这些token用空格连接，生成txt文件，最后用wordcloud生成
# need, toker = get_info_ranking_by_posibility()
# need500 = need[:500]
# TODO: 去除-1的项
def get_tokens_rank_10_each_case(need, toker):
    tokens = []
    for result, case, atts, idss in need:
        atts = np.array(atts)
        ranked_ids = (-atts).argsort()
        if len(ranked_ids) < 5: 
            pass
        else:
            for idx in ranked_ids[:10]:
                tokens.append(idss[idx])
    return tokens

def get_tokens(need, toker):
    tokens = get_tokens_rank_10_each_case(need, toker)
    tokens = [toker.decode(token_id).replace(' ', '') for token_id in tokens]
    return tokens

def run(need, toker):
    tokens = get_tokens(need, toker)
    text = ' '.join(tokens)
    wc = WordCloud(width=480, height=320, background_color="white", font_path=font)
    wc.generate(text)
    wc.to_file('wc2.png')
    
