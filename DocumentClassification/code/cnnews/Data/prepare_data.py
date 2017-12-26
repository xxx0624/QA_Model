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
dev_emb_file = 'cnews.dev.emb'
dev_tag_emb_file = 'cnews.dev.tag.emb'
dev_label_file = 'cnews.dev.label'

WORDS_DIM = 300
VEC_DIM = 300
UNKNOWN = 'UNKNOWN'
vectorsWordFile = 'word2vec.bin'
vectorsTagFile = 'tag_embedding.bin'


def load_vectors(vectorsBaikeFile):
    # w2v = Word2Vec.load_word2vec_format(vectorsBaikeFile, binary=True)
    w2v = KeyedVectors.load_word2vec_format(vectorsBaikeFile, binary=True)
    return w2v


def load_tag_vectors(vectorsfile):
    d = {}
    with codecs.open(vectorsfile, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            if line != '':
                index = line.index(' ')
                tag = line[:index]
                emb_list = []
                for emb in line[index+1:].split(','):
                    emb_list.append(float(emb))
                d[tag] = emb_list
    return d


def get_vector_of_dim(w2v, tag_w2v, word, vec_dim):
    if word == UNKNOWN:
        v_list = []
        for i in range(0, vec_dim):
            v_list.append(0.01)
        return v_list
    elif word in tag_w2v:
        #tag embedding
        v_list = tag_w2v[word]
        if vec_dim > len(v_list):
            for i in range(len(v_list), vec_dim, 1):
                v_list.append(np.random.uniform(-0.25,0.25))
        else:
            v_list = v_list[:vec_dim]
        return v_list
    elif word.decode('utf-8') in w2v.vocab:
        #word embedding
        v_list = w2v[word.decode('utf-8')].tolist()
        if vec_dim > len(v_list):
            for i in range(len(v_list), vec_dim, 1):
                v_list.append(np.random.uniform(-0.25,0.25))
        else:
            v_list = v_list[:vec_dim]
        return v_list
    else:
        v_list = []
        for i in range(0, vec_dim):
            v_list.append(np.random.uniform(-0.25,0.25))
        return v_list


def encode_sent(w2v, tag_w2v, sentence, vec_dim):
    if len(sentence) > WORDS_DIM:
        sentence = sentence[:WORDS_DIM]
    else:
        for i in range(len(sentence), WORDS_DIM, 1):
            sentence.append(UNKNOWN)
    x = []
    # sentence is a list [w1, w2, ...]
    for w in sentence:
        x.append(get_vector_of_dim(w2v, tag_w2v, w, vec_dim))
    return x


def get_label(word):
    word = str(word)
    label_list = {
        '体育': 0,
        '娱乐': 1,
        '家居': 2,
        '房产': 3,
        '教育': 4,
        '时尚': 5,
        '时政': 6,
        '游戏': 7,
        '科技': 8,
        '财经': 9
    }
    if word in label_list:
        return label_list[word]
    else:
        return -1


w2v = load_vectors(vectorsWordFile)
tag_w2v = load_tag_vectors(vectorsTagFile)

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
            index = line.find('\t')
            if index == -1:
                continue
            label = line[:index]
            label = get_label(label)
            if label == -1:
                raise Exception('train label...' + str(line))
            fw_label.write(str(label) + '\n')
            line = line[index+1:]
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
            words_emb = encode_sent(w2v, tag_w2v, word_list, VEC_DIM)
            tags_emb = encode_sent(w2v, tag_w2v, tag_list, VEC_DIM)
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
            index = line.find('\t')
            if index == -1:
                continue
            label = line[:index]
            label = get_label(label)
            if label == -1:
                raise Exception('test label...' + str(line))
            fw_label.write(str(label) + '\n')
            line = line[index+1:]
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
            words_emb = encode_sent(w2v, tag_w2v, word_list, VEC_DIM)
            tags_emb = encode_sent(w2v, tag_w2v, tag_list, VEC_DIM)
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
            index = line.find('\t')
            if index == -1:
                continue
            label = line[:index]
            label = get_label(label)
            if label == -1:
                raise Exception('dev label...' + str(line))
            fw_label.write(str(label) + '\n')
            line = line[index+1:]
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
            words_emb = encode_sent(w2v, tag_w2v, word_list, VEC_DIM)
            tags_emb = encode_sent(w2v, tag_w2v, tag_list, VEC_DIM)
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
