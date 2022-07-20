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
            ss.append(no_indicator(s)) # NOTE: 一定要去掉段落开头指示器
        loader.append((ss, labels))
    assert len(loader) == (length_max - window_size + 1)
    return loader

def read_ld_train():
    return create_loader(read_trains(), 4)

def read_ld_test():
    return create_loader(read_tests(), 4)


def read_ld_dev():
    return create_loader(read_devs(), 4)

# =============================================================================

# NOTE: 要考虑章节接续点
def read_ld_train_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_trains_from_chapters(), window_size)

def read_ld_test_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_tests_from_chapters(), window_size)

def read_ld_dev_from_chapters(window_size = 4):
    return create_loader_from_chapters(read_devs_from_chapters(), window_size)

# NOTE: 要考虑章节接续点
# NOTE: TESTED, window_size = 6的情况也已经测试 
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


def test_create_loader_from_chapters():
    ld = read_ld_dev_from_chapters(window_size = 6)
    print(ld[37])
    print(ld[38])
    print(ld[-1])
    

