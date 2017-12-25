import sys, codecs, os

reload(sys)
sys.setdefaultencoding('utf-8')

import jieba.posseg as pseg
import codecs
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors


def get_sentence(train_data, test_data, dev_data, sentence_file, tag_embedding_file):
    fw = codecs.open(sentence_file, 'w', encoding='utf-8')
    sentence = []
    tag_list = []
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
                for w in words:
                    wlist.append(w.word)
                    fw.write(w.word + ' ')
                    if not w.flag in tag_list:
                        tag_list.append(w.flag)
                sentence.append(wlist)
                fw.write('\n')
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
                for w in words:
                    wlist.append(w.word)
                    fw.write(w.word + ' ')
                    if not w.flag in tag_list:
                        tag_list.append(w.flag)
                sentence.append(wlist)
                fw.write('\n')
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
                for w in words:
                    wlist.append(w.word)
                    fw.write(w.word + ' ')
                    if not w.flag in tag_list:
                        tag_list.append(w.flag)
                sentence.append(wlist)
                fw.write('\n')
    fw.close()
    #generate tag embedding
    #(None, 300)
    tag_emb = np.random.normal(size=(len(tag_list), 300))
    fw = codecs.open(tag_embedding_file, 'w', encoding='utf-8')
    for i in range(len(tag_list)):
        fw.write(str(tag_list[i])+' ')
        for j in range(len(tag_emb[0])):
            if j == 0:
                fw.write(str(tag_emb[i][j]))
            else:
                fw.write(',' + str(tag_emb[i][j]))
        fw.write('\n')
    fw.close()
    return sentence


def train():
    train_data = 'cnews.train.txt'
    test_data = 'cnews.test.txt'
    dev_data = 'cnews.val.txt'
    out_path = 'word2vec.bin'
    sentence_file = 'all_sentence.txt'
    tag_embedding_file = 'tag_embedding.bin'
    model = Word2Vec(sg=1,
                     sentences=get_sentence(train_data, test_data, dev_data, sentence_file, tag_embedding_file),
                     size=300, window=5, min_count=3, workers=4, iter=50)
    model.wv.save_word2vec_format(out_path, binary=True)


def get_w2v():
    model = KeyedVectors.load_word2vec_format('word2vec.bin', binary=True)
    return model


if __name__ == "__main__":
    train()

    get_w2v()
