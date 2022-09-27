import itertools

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


