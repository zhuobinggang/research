from mainichi import *
import random
import data_jap_reader as data

def get_one_art_per_line(structed_articles):
    one_art_per_line = []
    for art in structed_articles:
        lines = [line.replace('$', '') for line in art]
        art_line = '$'.join(lines)
        one_art_per_line.append(art_line)
    return one_art_per_line

def customize_my_dataset_and_save(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    train = one_art_per_line[2000:4000]
    test = one_art_per_line[4000:4500]
    dev = one_art_per_line[4500:5000]
    manual_exp = one_art_per_line[5000:5500]
    # valid = one_art_per_line[2000:2500]
    with open('datasets/train.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(train))
    with open('datasets/test.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(test))
    with open('datasets/dev.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(dev))
    with open('datasets/manual_exp.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(manual_exp))

def write_additional_test_datasets(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    tests = []
    tests.append(one_art_per_line[5500:6000])
    tests.append(one_art_per_line[6000:6500])
    tests.append(one_art_per_line[6500:7000])
    tests.append(one_art_per_line[7000:7500])
    tests.append(one_art_per_line[7500:8000])
    tests.append(one_art_per_line[8500:9000])
    tests.append(one_art_per_line[9000:9500])
    tests.append(one_art_per_line[9500:10000])
    tests.append(one_art_per_line[10000:10500])
    tests.append(one_art_per_line[10500:11000])
    for i, test_ds in enumerate(tests):
        with open(f'datasets/test{i}.paragraph.txt', 'w') as the_file:
            the_file.write('\n'.join(test_ds))

def ld_without_opening(ld):
    ld = [case for case in ld if case[0][2] != 0]
    return ld

def read_additional_test_ds():
    tlds = []
    for i in range(10):
        tld = load_customized_loader(file_name = f'test{i}', half = 2, batch = 1, shuffle = False)
        tld = ld_without_opening(tld)
        tlds.append(tld)
    return tlds

def only_label_without_opening(name):
    tld = load_customized_loader(file_name = name, half = 2, batch = 1, shuffle = False)
    tld = ld_without_opening(tld)
    return [case[0][1][case[0][2]] for case in tld]

def read_additional_test_dataset_targets():
    ress = []
    for i in range(10):
        ress.append(only_label_without_opening(f'test{i}'))
    return ress

def read_train_dev_targets():
    return only_label_without_opening('train'), only_label_without_opening('dev')


def customize_my_dataset_and_save_mini(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    train = one_art_per_line[0:300]
    test = one_art_per_line[300:450]
    dev = one_art_per_line[450:600]
    valid = one_art_per_line[600:750]
    with open('datasets/train.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(train))
    with open('datasets/test.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(test))
    with open('datasets/dev.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(dev))
    with open('datasets/valid.paragraph.mini.txt', 'w') as the_file:
        the_file.write('\n'.join(valid))

def read_sentences_per_art(path):
    with open(path, 'r') as the_file:
        lines = the_file.readlines()
    arts = [line.split('$') for line in lines]
    return arts

# 读取存好的数据集然后读成loader
# 想想有什么简单的处理方式，没准要增加一个嵌套loader
# 不不对，之前loader里存的数据不也是处理过的吗？在处理的时候做点手脚就完了
def load_customized_loader_org(file_name = 'train', half = 2, shuffle = True, mini = False):
    if mini:
        arts = read_sentences_per_art(f'datasets/{file_name}.paragraph.mini.txt')
    else:
        arts = read_sentences_per_art(f'datasets/{file_name}.paragraph.txt')
    masses = []
    for art in arts:
        ds = data.Dataset(ss_len = half * 2, datas = art)
        ld = data.Loader_Symmetry(ds = ds, half = half, batch = 1) # 这里的batch不一样
        for mass in ld:
            ss, labels, pos = mass[0]
            masses.append((ss, labels, pos))
    if shuffle:
        random.shuffle(masses)
    else:
        pass
    return masses

def load_customized_loader(file_name = 'train', half = 2, batch = 1, shuffle = True, mini = False):
    masses = load_customized_loader_org(file_name, half, shuffle, mini)
    if batch == 1:
        # print('Warning: batch != 1 is not supported now')
        wrapped_batch = [[mess] for mess in masses]
        return wrapped_batch
    elif batch == 2:
        x = masses[0::2]
        y = masses[1::2]
        wrapped_batch = list(zip(x, y))
        return wrapped_batch
    elif batch == 3:
        x = masses[0::3]
        y = masses[1::3]
        z = masses[2::3]
        wrapped_batch = list(zip(x, y, z))
        return wrapped_batch
    elif batch == 4:
        x = masses[0::4]
        y = masses[1::4]
        z = masses[2::4]
        v = masses[3::4]
        wrapped_batch = list(zip(x, y, z, v))
        return wrapped_batch
    else:
        print('Fuck you, no more batch!')


# =============== Analysis Methods ==================

def cal_para_count(arts):
    counts = []
    for sentences in arts:
        counts.append(sum([1 for s in sentences if s.startswith('\u3000')]))
    return counts


# =============== 全新数据集创建逻辑 ================

def is_begining(s):
  return s.startswith('\u3000')

def no_indicator(s):
  return s.replace('\u3000', '')

def read_chapters(file_name = 'train'):
    arts = read_sentences_per_art(f'datasets/{file_name}.paragraph.txt')
    arts_without_linebreak = []
    for art in arts:
        art = [line.replace(' ', '').replace('\n', '').replace('\r', '') for line in art]
        arts_without_linebreak.append(art)
    return arts_without_linebreak

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

def read_ld_train(window_size = 4):
    return create_loader_from_chapters(read_chapters('train'), window_size)

def read_ld_test(window_size = 4):
    return create_loader_from_chapters(read_chapters('test'), window_size)

def read_ld_tests(window_size = 4):
    lds = []
    for i in range(10):
        lds.append(create_loader_from_chapters(read_chapters(f'test{i}'), window_size))
    return lds

def read_ld_dev(window_size = 4):
    return create_loader_from_chapters(read_chapters('dev'), window_size)
