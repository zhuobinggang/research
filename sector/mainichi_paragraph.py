from mainichi import *

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


