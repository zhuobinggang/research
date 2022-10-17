import random
import numpy as np
import re

END_CHARACTERS = ['。', '？']

def read_docs(data_id = 1):
  if data_id == 1:
    filenames = ['sansirou', 'sorekara', 'mon', 'higan', 'gyoujin']
  elif data_id == 2:
    filenames = ['kokoro']
  elif data_id == 3:
    filenames = ['meian']
  paths = [f'../sector/datasets/{name}_new.txt' for name in filenames]
  docs = []
  for path in paths:
    with open(path) as f:
      lines = f.readlines()
      docs.append(lines)
  return docs

def read_lines(data_id = 1):
  docs = read_docs(data_id)
  lines = []
  for doc in docs:
    lines += doc
  return [line.replace(' ', '').replace('\n', '').replace('\r', '') for line in lines] # 去掉空格和换行符

# 需要根据空行分割成不同的章节
def read_chapters(data_id = 1):
  lines = read_lines(data_id) # 隐式根据换行划分句子
  chapters = []
  sentences = []
  for line in lines:
      if len(line) == 0: # 空行
          chapters.append(sentences.copy())
          sentences = []
      else:
          s = ''
          for c in line: # 遍历行内所有character
              s += c
              if c in END_CHARACTERS:
                  sentences.append(s)
                  s = ''
          if len(s) > 0: # 处理最后留下来的一点
              if len(s) < 3:
                  sentences[-1] += s
              else:
                  sentences.append(s)
  if len(sentences) > 0:
      print('似乎出了点意外情况，理论上EOF应该是一个空行，所以不应该剩下这个才对')
  return chapters

# ===================

def create_loader_from_chapters(chapters, window_size = 4):
    loader = []
    assert window_size % 2 == 0
    half_window_size = int(window_size / 2)
    for sentences in chapters:
        length = len(sentences)
        end = len(sentences)
        for center_idx in range(1, length):
            ss = []
            labels = []
            for idx in range(center_idx - half_window_size, center_idx + half_window_size): # idx = 2 时候 range(0, 4)
                if idx < 0 or idx >= length:
                    labels.append(None) # NOTE: 一定要handle None
                    ss.append(None)
                else:
                    s = sentences[idx]
                    labels.append(1 if is_begining(s) else 0)
                    ss.append(no_indicator(s)) # NOTE: 一定要去掉段落开头指示器
            loader.append((ss, labels))
    # NOTE: ASSERT
    count = sum([len(sentences) - 1 for sentences in chapters])
    assert len(loader) == count
    return loader

def read_trains_from_chapters():
  return read_chapters(1)

def read_tests_from_chapters():
  return read_chapters(3)

def read_devs_from_chapters():
  return read_chapters(2)

def no_indicator(s):
  return s.replace('\u3000', '')

def is_begining(s):
  return s.startswith('\u3000')

def cal_label_one(ss):
  return sum([1 for s in ss if is_begining(s)])

# NOTE: 要考虑章节接续点
def read_ld_train_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_trains_from_chapters(), window_size)

def read_ld_test_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_tests_from_chapters(), window_size)

# NOTE: 用来计算t值
# 读取188个章节，每个章节一个loader
def read_lds_test_from_chapters(window_size = 4):
    chapters = read_tests_from_chapters()
    lds = []
    for chapter in chapters:
        lds.append(create_loader_from_chapters([chapter], window_size))
    return lds

def read_ld_dev_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_devs_from_chapters(), window_size)
