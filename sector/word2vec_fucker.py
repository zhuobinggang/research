from wikipedia2vec import Wikipedia2Vec
import unidic_lite
import MeCab

tagger = MeCab.Tagger(unidic_lite.DICDIR)
MODEL_FILE = '/home/taku/projects/jawiki_20180420_300d.pkl'
wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

def sentence_to_words(s):
  words = []
  node = tagger.parseToNode(s)
  while node is not None:
    #if node.feature.startswith('名詞'):
    words.append(node.surface)
    node = node.next
  return [word for word in words if len(word) > 0]

def sentence_to_wordvecs(s):
  words = sentence_to_words(s)
  vecs = []
  for word in words:
    try:
      vecs.append(wiki2vec.get_word_vector(word))
    except:
      #print(f'No word vec of {word}')
      pass
  return vecs
        

