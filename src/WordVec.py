#encoding=utf-8
from gensim.models import word2vec
sentences=word2vec.Text8Corpus('../data/train.txt')
model=word2vec.Word2Vec(sentences, size=200)
model.save('../wordvec200/word2vec.model')
