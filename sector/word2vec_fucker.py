from wikipedia2vec import Wikipedia2Vec
import unidic_lite
import MeCab

tagger = MeCab.Tagger(unidic_lite.DICDIR)
MODEL_FILE = '/usr/src/taku/jawiki_20180420_300d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

def sentence_to_words(s, max_len = 512):
  words = []
  node = tagger.parseToNode(s)
  while node is not None:
    #if node.feature.startswith('名詞'):
    words.append(node.surface)
    node = node.next
  return [word for word in words if len(word) > 0][:max_len]

def sentence_to_wordvecs(s, max_len = 512, require_words = False):
  words = sentence_to_words(s, max_len)
  vecs = []
  words_per_sentence = []
  for word in words:
    current_length = len(vecs)
    try:
      vecs.append(wiki2vec.get_word_vector(word))
    except:
      #print(f'No word vec of {word}')
      pass
    if current_length != len(vecs): # 增加了
      words_per_sentence.append(word)
    else:
      pass
  if require_words:
    return vecs, words_per_sentence
  else:
    return vecs 

