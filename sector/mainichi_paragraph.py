from mainichi import *
import random
import data_jap_reader as data

def customize_my_dataset_and_save(structed_articles):
    one_art_per_line = []
    for art in structed_articles:
        art = [line.replace('$', '') for line in art]
        line = '$'.join(art)
        one_art_per_line.append(line)
    train = one_art_per_line[0:2000]
    test = one_art_per_line[2000:3000]
    dev = one_art_per_line[3000:4000]
    valid = one_art_per_line[4000:5000]
    with open('datasets/train.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(train))
    with open('datasets/test.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(test))
    with open('datasets/dev.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(dev))
    with open('datasets/valid.paragraph.txt', 'w') as the_file:
        the_file.write('\n'.join(valid))


# TODO: 读取存好的数据集然后读成loader
# 想想有什么简单的处理方式，没准要增加一个嵌套loader
# 不不对，之前loader里存的数据不也是处理过的吗？在处理的时候做点手脚就完了
def load_customized_dataset(file_name = 'train', half = 2, shuffle = True):
    with open(f'datasets/{file_name}.paragraph.txt', 'r') as the_file:
        lines = the_file.readlines()
    arts = [line.split('$') for line in lines]
    masses = []
    for art in arts:
        ds = data.Dataset(ss_len = half * 2, datas = art)
        ld = data.Loader_Symmetry(ds = ds, half = 2, batch = 1)
        for mass in ld:
            ss, labels, pos = mass[0]
            masses.append((ss, labels, pos))
    if shuffle:
        random.shuffle(masses)
    else:
        pass
    return masses


