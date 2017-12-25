#  -*- coding: utf-8 -*-
import sys

reload(sys)
sys.setdefaultencoding('utf8')

import codecs
import numpy as np
# from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import jieba.posseg as pseg

train_file = 'cnews.train.txt'
test_file = 'cnews.test.txt'
dev_file = 'cnews.val.txt'
train_emb_file = 'cnews.train.emb'
train_tag_emb_file = 'cnews.train.tag.emb'
train_label_file = 'cnews.train.label'
test_emb_file = 'cnews.test.emb'
test_tag_emb_file = 'cnews.test.tag.emb'
test_label_file = 'cnews.test.label'
dev_emb_file = 'cnews.test.emb'
dev_tag_emb_file = 'cnews.test.tag.emb'
dev_label_file = 'cnews.test.label'

WORDS_DIM = 300
VEC_DIM = 300
UNKNOWN = 'UNKNOWN'
vectorsBaikeFile = '/opt/exp_data/word_vector_cn/baike_vector.bin'
vectorsTagFile = 'word2vec.bin'


def load_vectors(vectorsBaikeFile):
    # w2v = Word2Vec.load_word2vec_format(vectorsBaikeFile, binary=True)
    w2v = KeyedVectors.load_word2vec_format(vectorsBaikeFile, binary=True)
    return w2v


def load_tag_w2v(vectorsTagFile):
    w2v = KeyedVectors.load_word2vec_format(vectorsTagFile, binary=True)
    return w2v


def get_vector_of_dim(w2v, word, vec_dim):
    if word == UNKNOWN:
        v_list = []
        for i in range(0, vec_dim):
            v_list.append(0.01)
        return v_list
    elif word.decode('utf-8') in w2v.vocab:
        v_list = w2v[word.decode('utf-8')].tolist()
        if vec_dim > len(v_list):
            for i in range(len(v_list), vec_dim, 1):
                v_list.append(0.01)
        else:
            v_list = v_list[:vec_dim]
        return v_list
    else:
        v_list = []
        for i in range(0, vec_dim):
            v_list.append(0.01)
        return v_list


def encode_sent(w2v, sentence, vec_dim):
    if len(sentence) > WORDS_DIM:
        sentence = sentence[:WORDS_DIM]
    else:
        for i in range(len(sentence), WORDS_DIM, 1):
            sentence.append(UNKNOWN)
    x = []
    # sentence is a list [w1, w2, ...]
    for w in sentence:
        x.append(get_vector_of_dim(w2v, w, vec_dim))
    return x


def label(word):
    label_list = {
        '体育': 1,
        '娱乐': 1,
        '家居': 1,
        '房产': 1,
        '教育': 1,
        '时尚': 1,
        '时政': 1,
        '游戏': 1,
        '科技': 1,
        '财经': 1
    }
    if word in label_list:
        return label_list[word]
    else:
        return -1


# w2v_baike = load_tag_w2v(vectorsBaikeFile)
w2v_tag = load_tag_w2v(vectorsTagFile)

with codecs.open(train_file, 'r', encoding='utf-8') as fr:
    fw_word = codecs.open(train_emb_file, 'w', encoding='utf-8')
    fw_tag = codecs.open(train_tag_emb_file, 'w', encoding='utf-8')
    fw_label = codecs.open(train_label_file, 'w', encoding='utf-8')
    line_cnt = 0
    word_cnt = 0
    for line in fr:
        line = line.strip()
        print '[', line_cnt, '] prepare train data...'
        if line != "":
            label = line.split("\t")[0]
            fw_label.write(str(label) + '\n')
            line = line.split("\t")[1]
            words = pseg.cut(line)
            line_cnt += 1
            id = 0
            word_list = []
            tag_list = []
            for w in words:
                # print w.word, w.flag
                word_list.append(w.word)
                tag_list.append(w.flag)
                id += 1
            words_emb = encode_sent(w2v_tag, word_list, VEC_DIM)
            tags_emb = encode_sent(w2v_tag, tag_list, VEC_DIM)
            for i in range(len(words_emb)):
                for j in range(len(words_emb[0])):
                    if j == 0:
                        fw_word.write(str(words_emb[i][j]))
                        fw_tag.write(str(tags_emb[i][j]))
                    else:
                        fw_word.write(',' + str(words_emb[i][j]))
                        fw_tag.write(',' + str(tags_emb[i][j]))
                fw_word.write('\n')
                fw_tag.write('\n')
            word_cnt += id
    fw_word.close()
    fw_label.close()
    fw_tag.close()
    print 'avg = ', word_cnt / line_cnt

with codecs.open(test_file, 'r', encoding='utf-8') as fr:
    fw_word = codecs.open(test_emb_file, 'w', encoding='utf-8')
    fw_tag = codecs.open(test_tag_emb_file, 'w', encoding='utf-8')
    fw_label = codecs.open(test_label_file, 'w', encoding='utf-8')
    line_cnt = 0
    word_cnt = 0
    for line in fr:
        line = line.strip()
        print '[', line_cnt, '] prepare test data...'
        if line != "":
            label = line.split("\t")[0]
            fw_label.write(str(label) + '\n')
            line = line.split("\t")[1]
            words = pseg.cut(line)
            line_cnt += 1
            id = 0
            word_list = []
            tag_list = []
            for w in words:
                # print w.word, w.flag
                word_list.append(w.word)
                tag_list.append(w.flag)
                id += 1
            words_emb = encode_sent(w2v_tag, word_list, VEC_DIM)
            tags_emb = encode_sent(w2v_tag, tag_list, VEC_DIM)
            for i in range(len(words_emb)):
                for j in range(len(words_emb[0])):
                    if j == 0:
                        fw_word.write(str(words_emb[i][j]))
                        fw_tag.write(str(tags_emb[i][j]))
                    else:
                        fw_word.write(',' + str(words_emb[i][j]))
                        fw_tag.write(',' + str(tags_emb[i][j]))
                fw_word.write('\n')
                fw_tag.write('\n')
            word_cnt += id
    fw_word.close()
    fw_label.close()
    fw_tag.close()
    print 'avg = ', word_cnt / line_cnt

with codecs.open(dev_file, 'r', encoding='utf-8') as fr:
    fw_word = codecs.open(dev_emb_file, 'w', encoding='utf-8')
    fw_tag = codecs.open(dev_tag_emb_file, 'w', encoding='utf-8')
    fw_label = codecs.open(dev_label_file, 'w', encoding='utf-8')
    line_cnt = 0
    word_cnt = 0
    for line in fr:
        line = line.strip()
        print '[', line_cnt, '] prepare dev data...'
        if line != "":
            label = line.split("\t")[0]
            fw_label.write(str(label) + '\n')
            line = line.split("\t")[1]
            words = pseg.cut(line)
            line_cnt += 1
            id = 0
            word_list = []
            tag_list = []
            for w in words:
                # print w.word, w.flag
                word_list.append(w.word)
                tag_list.append(w.flag)
                id += 1
            words_emb = encode_sent(w2v_tag, word_list, VEC_DIM)
            tags_emb = encode_sent(w2v_tag, tag_list, VEC_DIM)
            for i in range(len(words_emb)):
                for j in range(len(words_emb[0])):
                    if j == 0:
                        fw_word.write(str(words_emb[i][j]))
                        fw_tag.write(str(tags_emb[i][j]))
                    else:
                        fw_word.write(',' + str(words_emb[i][j]))
                        fw_tag.write(',' + str(tags_emb[i][j]))
                fw_word.write('\n')
                fw_tag.write('\n')
            word_cnt += id
    fw_word.close()
    fw_label.close()
    fw_tag.close()
    print 'avg = ', word_cnt / line_cnt
