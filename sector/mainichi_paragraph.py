from mainichi import *
import random
import data_jap_reader as data

def get_one_art_per_line(structed_articles):
    one_art_per_line = []
    for art in structed_articles:
        art = [line.replace('$', '') for line in art]
        line = '$'.join(art)
        one_art_per_line.append(line)
    return one_art_per_line

def customize_my_dataset_and_save(structed_articles):
    one_art_per_line = get_one_art_per_line(structed_articles)
    train = one_art_per_line[0:1000]
    test = one_art_per_line[1000:1500]
    dev = one_art_per_line[1500:2000]
    valid = one_art_per_line[2000:2500]
    with open('datasets/train.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(train))
    with open('datasets/test.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(test))
    with open('datasets/dev.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(dev))
    with open('datasets/valid.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(valid))

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


