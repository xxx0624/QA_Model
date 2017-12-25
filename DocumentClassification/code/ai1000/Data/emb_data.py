import sys

reload(sys)
sys.setdefaultencoding('utf8')

import codecs
import numpy as np
# from gensim.models import Word2Vec
from gensim.models import KeyedVectors

WORDS_DIM = 150
VEC_DIM = 300
UNKNOWN = 'UNKNOWN'
vectorsWordFile = 'word2vec.bin'
vectorsTagFile = 'tag_embedding.bin'


def load_vectors(vectorsfile):
    # w2v = Word2Vec.load_word2vec_format(vectorsfile, binary=True)
    w2v = KeyedVectors.load_word2vec_format(vectorsfile, binary=True)
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
    x = []
    # sentence is a list [w1, w2, ...]
    for w in sentence:
        x.append(get_vector_of_dim(w2v, tag_w2v, w, vec_dim))
    return x


w2v = load_vectors(vectorsWordFile)
tag_w2v = load_tag_vectors(vectorsTagFile)
# generage train data
# line1:0,1,2,3...(VEC_DIM)
# line2:1,2,3,4...(VEC_DIM)
fw_w = codecs.open('training.seg.word.emb', 'w', encoding='utf-8')
fw_tag = codecs.open('training.seg.tag.emb', 'w', encoding='utf-8')
fw_label = codecs.open('training.seg.label.emb', 'w', encoding='utf-8')
# count lines
line_cnt = 0
with codecs.open('training.seg.csv', 'r', encoding='utf-8') as fr:
    for line in fr:
        line = line.strip()
        if line != '':
            line_cnt += 1
    print line_cnt
id = 0
with codecs.open('training.seg.csv', 'r', encoding='utf-8') as fr:
    for line in fr:
        id += 1
        print id
        line = line.strip()
        if line != '':
            line = line.split(',')
            if len(line) == 2:
                label = line[0]
                _words = line[1].strip().split(' ')
                fw_label.write(label)
                fw_label.write('\n')
                w_list = []
                tag_list = []
                cnt = 0
                for _w in _words:
                    w = _w.strip().split('/')
                    w_list.append(w[0])
                    tag_list.append(w[1])
                    cnt += 1
                    if cnt >= 150:
                        break
                for i in range(cnt, WORDS_DIM, 1):
                    w_list.append(UNKNOWN)
                    tag_list.append(UNKNOWN)
                for i in range(len(w_list)):
                    w_vec = get_vector_of_dim(w2v, tag_w2v, w_list[i], VEC_DIM)
                    tag_vec = get_vector_of_dim(w2v, tag_w2v, tag_list[i], VEC_DIM)
                    for item in w_vec:
                        fw_w.write(str(item) + ',')
                    fw_w.write('\n')
                    for item in tag_vec:
                        fw_tag.write(str(item) + ',')
                    fw_tag.write('\n')
            else:
                print 'error......'
fw_w.close()
fw_label.close()
fw_tag.close()


# generate test data
# line1:0,1,2,3...(VEC_DIM)
# line2:1,2,3,4...(VEC_DIM)
fw_w = codecs.open('testing.seg.word.emb', 'w', encoding='utf-8')
fw_tag = codecs.open('testing.seg.tag.emb', 'w', encoding='utf-8')
fw_label = codecs.open('testing.seg.label.emb', 'w', encoding='utf-8')
# count lines
line_cnt = 0
with codecs.open('testing.seg.csv', 'r', encoding='utf-8') as fr:
    for line in fr:
        line = line.strip()
        if line != '':
            line_cnt += 1
    print line_cnt
id = 0
with codecs.open('testing.seg.csv', 'r', encoding='utf-8') as fr:
    for line in fr:
        id += 1
        print id
        line = line.strip()
        if line != '':
            line = line.split(',')
            if len(line) == 2:
                label = line[0]
                _words = line[1].strip().split(' ')
                fw_label.write(label)
                fw_label.write('\n')
                w_list = []
                tag_list = []
                cnt = 0
                for _w in _words:
                    w = _w.strip().split('/')
                    w_list.append(w[0])
                    tag_list.append(w[1])
                    cnt += 1
                    if cnt >= 150:
                        break
                for i in range(cnt, WORDS_DIM, 1):
                    w_list.append(UNKNOWN)
                    tag_list.append(UNKNOWN)
                for i in range(len(w_list)):
                    w_vec = get_vector_of_dim(w2v, tag_w2v, w_list[i], VEC_DIM)
                    tag_vec = get_vector_of_dim(w2v, tag_w2v, tag_list[i], VEC_DIM)
                    for item in w_vec:
                        fw_w.write(str(item) + ',')
                    fw_w.write('\n')
                    for item in tag_vec:
                        fw_tag.write(str(item) + ',')
                    fw_tag.write('\n')
            else:
                print 'error......'
fw_w.close()
fw_label.close()
fw_tag.close()
