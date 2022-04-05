# 1: 选出最受瞩目的token，用wordcloud来展示
# 2: 用谷歌浏览器渲染
from sec_paragraph import get_att_ours, get_att_baseline
from exp7 import load_mld
import torch as t
import numpy as np
font = '/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf'
# font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
from wordcloud import WordCloud
from sec_paragraph import chrome_render_v3

def load_m(AUX=True):
    if AUX:
        return t.load('save/r01_fl50_0.tch')  
    else:
        return t.load('save/fl20_0_e3.tch')  

# 从mld选出取出正确率最高的100个例子
def get_info_ranking_by_posibility(AUX = True):
    m = load_m(AUX)
    pcases = cases_positive(load_mld())
    need = cases_ranking_from_max(m, pcases, AUX)
    return need, m.toker

def cases_positive(ld):
    results = []
    for case in ld:
        ss, labels, pos = case[0]
        if labels[pos] == 1:
            results.append(case)
    return results

def cases_ranking_from_max(m, cases, AUX = True):
    need = []
    for case in cases:
        if AUX:
            atts, idss, results, targets, labelss = get_att_ours(m, case)
        else:
            atts, idss, results, targets, labelss = get_att_baseline(m, case)
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

def case_checker(need, idx):
    res, case, att, ids = need[idx]
    print(f'{res}: {case}')
    

# 2: 用谷歌浏览器渲染
def run_chrome_render():
    idx = 16
    case_checker(need, idx)
    res, case, att, ids = need[idx]
    chrome_render_v3(toker, ids, att, labels = case[0][1], rank = idx, p = res)
