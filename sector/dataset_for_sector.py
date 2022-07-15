from data_jap_reader import *

# NOTE: 因为小说数据集只有头跟尾两个例外，直接无视特殊情况
def create_loader(sentences, window_size = 4):
    loader = []
    length_max = len(sentences)
    start = 0
    end = length_max - window_size + 1 # (+1代表包含)
    half_window = int(window_size / 2)
    for idx in range(start, end):
        ss = []
        labels = []
        for left_idx in range(idx, idx + window_size): # (0, 4)
            s = sentences[left_idx]
            labels.append(1 if is_begining(s) else 0)
            ss.append(no_indicator(s))
        loader.append((ss, labels))
    assert len(loader) == (length_max - window_size + 1)
    return loader

def read_ld_train():
    return create_loader(read_trains(), 4)

def read_ld_test():
    return create_loader(read_tests(), 4)


def read_ld_dev():
    return create_loader(read_devs(), 4)
