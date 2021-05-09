# 每日新闻处理脚本
from importlib import reload

the_path = './datasets/mai2019.utf8.txt'

def read_lines():
  with open(the_path) as fin:
    lines = fin.readlines()
  return lines
  
def get_articles_raw():
  articles = []
  lines = read_lines()
  starts = [idx for idx, line in enumerate(lines) if line.startswith('＼ＡＤ＼')]
  borders = list(zip(starts, starts[1:])) # 丢弃最后一个文本
  for start, end in borders:
    articles.append(lines[start: end])
  return articles

def filter1_start_line_raws(start_line): 
  return start_line.startswith('＼ＡＤ＼０１') or start_line.startswith('＼ＡＤ＼０２') or start_line.startswith('＼ＡＤ＼０３') or start_line.startswith('＼ＡＤ＼１０')
  
# 1. 将所有AD=01(1面), 02(2面), 03(3面), 10(特集)的去掉
def special_type_articles_filtered(articles):
  return [art for art in articles if not filter1_start_line_raws(art[0])]

def except_t2_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([line for line in art if line.startswith('＼Ｔ２＼')])
  return new_articles

def special_tokens_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([remove_special_tokens(line) for line in art])
  return new_articles

def line_without_period_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([line for line in art if line.find('。') >= 0])
  return new_articles

def remove_special_tokens(line):
  special_tokens = ['\u3000', '\n', '＼Ｔ２＼', '「', '」', '（','）', '○', '＜', '＞', '◆', '〓', '｝', '｛', '■']
  result = line
  for token in special_tokens:
    result = result.replace(token, '')
  return result

def paragraph_less_then_num_removed(articles, num = 2): 
  return [art for art in articles if not len(art) < num]

def build_structure(articles):
  new_articles = []
  for art in articles:
    sentences = []
    for paragrath in art:
      new_ss = paragrath.split('。')
      new_ss = new_ss[:-1] # 排除最后一个元素，排除像【ワシントン中井正裕】这种东西
      new_ss = [s for s in new_ss if len(s) > 1] # 排除长度过小的句子包括''
      new_ss = [s + '。' for s in new_ss] # 重新补充句号
      if len(new_ss) > 0:
        new_ss[0] = '\u3000' + new_ss[0] # 为了适应以前的数据集
        sentences += new_ss
      else:
        print(f'WARN: Empty paragrath! {paragrath}')
    new_articles.append(sentences)
  return new_articles
      
def line_with_special_token_removed(articles):
  new_articles = []
  for art in articles:
    new_articles.append([line for line in art if (line.find('】') == -1 and line.find('【') == -1 and line.find('＜') == -1 and line.find('＞') == -1 and line.find('◆') == -1)])
  return new_articles

def paragraph_only_one_sentence_removed(articles):
  results = []
  for art in articles:
    paras = []
    for para in art:
      if len(para.split('。')) <= 2:
        pass
      else:
        paras.append(para)
    results.append(paras)
  return results

def paragraph_with_special_token_removed(articles):
  results = []
  for art in articles:
    paras = [para for para in art if para.find('【') == -1]
    results.append(paras)
  return results

def standard_process():
  articles = get_articles_raw()
  articles = special_type_articles_filtered(articles) # NO.1
  articles = except_t2_removed(articles)
  articles = special_tokens_removed(articles)
  articles = line_without_period_removed(articles) # NO.2
  # articles = line_with_special_token_removed(articles) # NO.2
  articles = paragraph_only_one_sentence_removed(articles) # NO.2
  articles = paragraph_less_then_num_removed(articles, num = 2) # NO.3
  articles = paragraph_with_special_token_removed(articles) # NO.2
  articles = build_structure(articles)
  return articles

# output = 5
def avg_sentence_len(articles):
  lengths = [len(art) for art in articles]
  return int(sum(lengths) / len(lengths))
    
def save_origin_train_ds(structed_articles):
  flated = [item for sublist in structed_articles for item in sublist]
  train = flated[:10000]
  test = flated[10000:15000]
  with open('train.txt', 'a') as the_file:
    for line in train:
      the_file.write(f'{line}\n')
  with open('test.txt', 'a') as the_file:
    for line in test:
      the_file.write(f'{line}\n')

def no_line_breaks(texts):
  return [text.replace('\n', '') for text in texts]

def read_trains(mini = False):
  file_path = 'train.mini.txt' if mini else 'train.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_tests(mini = False):
  file_path = 'test.mini.txt' if mini else 'test.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_valid(mini = False):
  file_path = 'valid.mini.txt' if mini else 'valid.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_trains_big(mini = False):
  file_path = 'train.big.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines

def read_tests_big(mini = False):
  file_path = 'test.big.txt'
  with open(file_path, encoding="utf8", errors='ignore') as the_file:
    lines = no_line_breaks(the_file.readlines())
  return lines
