import sys, codecs, os

reload(sys)
sys.setdefaultencoding('utf-8')

import jieba.posseg as pseg
import codecs
import pickle
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def get_sentence(train_data, test_data, dev_data):
    sentence = []
    with codecs.open(train_data, 'r', encoding='utf-8') as fr:
        id = 0
        for line in fr:
            print '[', id, ']...train...'
            id += 1
            line = line.strip()
            if line != "":
                index = line.find('\t')
                if index == -1:
                    continue
                # label = line.split("\t")[0]
                line = line[index+1:]
                words = pseg.cut(line)
                wlist = []
                tlist = []
                for w in words:
                    wlist.append(w.word)
                    tlist.append(w.flag)
                sentence.append(wlist)
                sentence.append(tlist)
    with codecs.open(test_data, 'r', encoding='utf-8') as fr:
        id = 0
        for line in fr:
            print '[', id, ']...test...'
            id += 1
            line = line.strip()
            if line != "":
                index = line.find('\t')
                if index == -1:
                    continue
                # label = line.split("\t")[0]
                line = line[index+1:]
                words = pseg.cut(line)
                wlist = []
                tlist = []
                for w in words:
                    wlist.append(w.word)
                    tlist.append(w.flag)
                sentence.append(wlist)
                sentence.append(tlist)
    with codecs.open(dev_data, 'r', encoding='utf-8') as fr:
        id = 0
        for line in fr:
            print '[', id, ']...dev...'
            id += 1
            line = line.strip()
            if line != "":
                index = line.find('\t')
                if index == -1:
                    continue
                # label = line.split("\t")[0]
                line = line[index+1:]
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
    train_data = 'cnews.train.txt'
    test_data = 'cnews.test.txt'
    dev_data = 'cnews.val.txt'
    out_path = 'word2vec.bin'
    model = Word2Vec(sg=1,
                     sentences=get_sentence(train_data, test_data, dev_data),
                     size=300, window=5, min_count=3, workers=4, iter=50)
    model.wv.save_word2vec_format(out_path, binary=True)


def get_w2v():
    model = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
    return model


if __name__ == "__main__":
    train()

    get_w2v()
