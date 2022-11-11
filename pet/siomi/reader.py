import itertools
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from five_split_data import goods, bads

def flatten(lists):
  return [item for lst in lists for item in lst]

def read_five_splits():
    res = []
    for i in range(5):
        # Test
        index_test = i
        test_goods = goods[index_test]
        test_bads = bads[index_test]
        test_pairs = [(goodword, badword) for (goodword, _), (badword, _) in zip(test_goods, test_bads)]
        # Dev: the one next to test
        index_dev = (i + 1) % 5
        dev_goods = goods[index_dev]
        dev_bads = bads[index_dev]
        dev_pairs = [(goodword, badword) for (goodword, _), (badword, _) in zip(dev_goods, dev_bads)]
        # Train
        train_goods = [item for index, item in enumerate(goods) if index not in [index_test, index_dev]]
        train_bads = [item for index, item in enumerate(bads) if index not in [index_test, index_dev]]
        # Train: expect the last one
        train_goods = flatten(train_goods)
        train_bads = flatten(train_bads)
        train_pairs = [(goodword, badword) for (goodword, _), (badword, _) in zip(train_goods, train_bads)]
        # (train, test, dev)
        res.append((train_pairs, test_pairs, dev_pairs))
    return res

def read_data():
    f_i = open('text_bassui.txt', 'r', encoding='UTF-8')
    word1_l=[]#good
    word2_l=[]#bad
    for data in f_i:
     data = data.lstrip('□')#□を消す
     data = data.strip()#改行を消す
     #data = data.lstrip('□')
     data=data.split('→')#→を消してそこで区切る
     word1_l.append(data[1::2])
     word2_l.append(data[::2])
    word1_l = list(itertools.chain.from_iterable(word1_l))
    word2_l = list(itertools.chain.from_iterable(word2_l))
    f_i.close()
    return list(zip(word2_l, word1_l))

# return: good_words, bad_words
def read_regular_ds():
    fi_daka = open('kakudaka_277.txt', 'r', encoding='UTF-8')
    fi_hiku = open('kakuhiku_277.txt', 'r', encoding='UTF-8')
    for data in fi_daka:
        good_words =data.split('|')#|を消してそこで区切る
    for data in fi_hiku:
        bad_words=data.split('|')#|を消してそこで区切る
    fi_daka.close()
    fi_hiku.close()
    del good_words[-1]#末尾の空白を削除
    del bad_words[-1]
    return good_words, bad_words

def read_regular_ds_zip():
    good_words, bad_words = read_regular_ds()
    return list(zip(bad_words, good_words))


def shuffle(array):
    res = array.copy()
    np.random.seed(0)
    np.random.shuffle(res)
    return res
    
def customized_ds():
    good_words, bad_words = read_regular_ds()
    good_words = [(item, 1) for item in good_words]
    bad_words = [(item, 0) for item in bad_words]
    ds = shuffle([] + good_words + bad_words)
    train_ds = ds[:448] # 448
    test_ds = ds[448:] # 106
    return train_ds, test_ds

def calculate_result(pred_y, true_y = None):
    if true_y is None:
        _, test_ds = customized_ds()
        true_y = [label for text, label in test_ds]
    f = f1_score(true_y, pred_y, average='macro')
    prec = precision_score(true_y, pred_y, average='macro')
    rec = recall_score(true_y, pred_y, average='macro')
    print(f'F: {f}, PRECISION: {prec}, RECALL: {rec}')


