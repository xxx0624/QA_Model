import sys, codecs, os

reload(sys)
sys.setdefaultencoding('utf-8')

import jieba.posseg as pseg
import codecs
import pickle
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def get_sentence(train_data, test_data):
    sentence = []
    with codecs.open(train_data, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != "":
                label = line.split(",")[0]
                line = line.split(",")[1]
                words = pseg.cut(line)
                wlist = []
                tlist = []
                for w in words:
                    wlist.append(w.word)
                    tlist.append(w.flag)
                sentence.append(wlist)
                sentence.append(tlist)
    with codecs.open(test_data, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != "":
                label = line.split(",")[0]
                line = line.split(",")[1]
                words = pseg.cut(line)
                wlist = []
                tlist = []
                for w in words:
                    wlist.append(w.word)
                    tlist.append(w.flag)
                sentence.append(wlist)
                sentence.append(tlist)
    return sentence


def train():
    train_data = 'training.csv'
    test_data = 'testing.csv'
    out_path = 'word2vec.bin'
    model = Word2Vec(sg=1,
                     sentences=get_sentence(train_data, test_data),
                     size=256, window=5, min_count=3, workers=4, iter=40)
    model.wv.save_word2vec_format(out_path, binary=True)


def get_w2v(model, word):
    # model = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
    if word in model.vocab:
        print model[word]
    else:
        print '2333'


if __name__ == "__main__":
    model = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
    get_w2v(model, 'n')
