#!/usr/bin/python
#  -*- coding: utf-8 -*-


import time
import os, sys
import random
import operator

sys.path.append('/home/zhengxing/QAModel')

import numpy as np
import theano
from theano import config
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
#from gensim.models import Word2Vec
from gensim.models import KeyedVectors

from util.timeUtil import GetNowTime
from util.logUtil import log

theano.config.floatX = 'float32'

#TODO change filepath to your local environment
'''
#include train test1 vectors.nobin
trainfile = '../../corpus/nlpcc16-data/train_1_detail'
train0file = '../../corpus/nlpcc16-data/train_0_detail'
test1file = '../../corpus/nlpcc16-data/test'
vectorsfile = '/opt/exp_data/word_vector_cn/baike_vector.bin'
'''
version_num = '1'
trainfile = '/home/zhengxing/QAModel/corpus/nlpcc16-dbqa-data/backup/train1_data_version_' + version_num
train0file = '/home/zhengxing/QAModel/corpus/nlpcc16-dbqa-data/backup/train0_data_version_' + version_num
test1file = '/home/zhengxing/QAModel/corpus/nlpcc16-dbqa-data/test/test_data_version_' + version_num
vectorsfile = '/home/zhengxing/exp_data/word_vector_cn/baike_vector.bin'
#vectorsfile = '/opt/exp_data/word_vector_cn/all_dbqa.bin'

#log file
logfile_path = '/home/zhengxing/QAModel/log/nlpcc'


class myDict(object):
    def __init__(self, v_list):
        self.v = v_list


def load_train0_dict():
    global train0file
    train0Dict = {}
    with open(train0file) as f:
        for line in f:
            line = line.strip().split('\t')
            qid = int(line[1].split(":")[-1])
            if qid not in train0Dict:
                train0Dict[qid] = myDict([])
            train0Dict[qid].v.append(line[3])
    return train0Dict

def build_vocab():
    global trainfile
    global train0file
    code, vocab = int(0), {}
    vocab['UNKNOWN'] = code
    code += 1
    for line in open(train0file):
        items = line.strip().split('\t')
        for i in range(2, 4):
            for word in items[i].split('_'):
                if len(word) <= 0:
                    continue
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    for line in open(trainfile):
        items = line.strip().split('\t')
        for i in range(2, 4):
            for word in items[i].split('_'):
                if len(word) <= 0:
                    continue
                if not word in vocab:
                    vocab[word] = code
                    code += 1
    return vocab


def load_vectors():
    global vectorsfile
    #w2v = Word2Vec.load_word2vec_format(vectorsfile, binary=True)
    w2v = KeyedVectors.load_word2vec_format(vectorsfile, binary=True)
    return w2v


def get_vector_of_dim(w2v, word, dim):
    if word.decode('utf-8')  in w2v.vocab:
        v_list = w2v[word.decode('utf-8')].tolist()
        if dim > len(v_list):
            for i in range(len(v_list), dim, 1):
                v_list.append(0.01)
        else:
            v_list = v_list[:dim]
        return v_list
    else:
        v_list = []
        for i in range(0, dim):
            v_list.append(0.01)
        return v_list


def load_word_embeddings(vocab, embedding_size):
    w2v = load_vectors()
    embeddings = [] #brute initialization
    #print 'vocab size = ', len(vocab)
    for i in range(0, len(vocab)):
        vec = []
        for j in range(0, embedding_size):
            vec.append(0.01)
        embeddings.append(vec)
    #print 'embedding size = ', len(embeddings)
    #id = 0
    for word, code in vocab.items():
        #print 'id = ', id, 'code = ', code, 'word = ', word
        embeddings[code] = get_vector_of_dim(w2v, word, embedding_size)
    return np.array(embeddings, dtype=theano.config.floatX)


#be attention initialization of UNKNNOW
def encode_sent(vocab, string, size):
    x, m = [], []
    words = string.split('_')
    for i in range(0, size):
        if words[i] in vocab:
            x.append(vocab[words[i]])
        else:
            x.append(vocab['UNKNOWN'])
        if words[i] == '<a>': #TODO
            m.append(1) #fixed sequence length, else use 0
        else:
            m.append(1)
    return x, m


def load_train_list():
    global trainfile
    trainList = []
    for line in open(trainfile):
        trainList.append(line.strip().split('\t'))
    return trainList


def load_test_list():
    global test1file
    testList = []
    for line in open(test1file):
        testList.append(line.strip().split('\t'))
    return testList


def check_int(id):
    id = str(id)
    for i in id:
        if i < '0' or i > '9':
            return False
    return True


def load_train_data_from_2files(train0Dict, train1List, vocab, batch_size, words_num_dim):
    train_1, train_2, train_3 = [], [], []
    mask_1, mask_2, mask_3 = [], [], []
    counter = 0
    while True:
        index = random.randint(0, len(train1List)-1)
        pos = train1List[index]
        qid = int(pos[1].strip().split(":")[-1])
        if check_int(qid) == False:
            continue
        neg = None
        if qid in train0Dict:
            neg = train0Dict[qid].v[(random.randint(0, len(train0Dict[qid].v) - 1))]
        else:
            neg = train1List[random.randint(0, len(train1List)-1)][3]
        x, m = encode_sent(vocab, pos[2], words_num_dim)
        train_1.append(x)
        mask_1.append(m)
        x, m = encode_sent(vocab, pos[3], words_num_dim)
        train_2.append(x)
        mask_2.append(m)
        x, m = encode_sent(vocab, neg, words_num_dim)
        train_3.append(x)
        mask_3.append(m)
        counter += 1
        if counter >= batch_size:
            break
    return np.transpose(np.array(train_1, dtype=theano.config.floatX)), np.transpose(np.array(train_2, dtype=theano.config.floatX)), np.transpose(np.array(train_3, dtype=theano.config.floatX)), np.transpose(np.array(mask_1, dtype=theano.config.floatX)) , np.transpose(np.array(mask_2, dtype=theano.config.floatX)), np.transpose(np.array(mask_3, dtype=theano.config.floatX))


'''
def load_data(trainList, vocab, batch_size):
    global words_num_dim
    train_1, train_2, train_3 = [], [], []
    mask_1, mask_2, mask_3 = [], [], []
    counter = 0
    while True:
        pos = trainList[random.randint(0, len(trainList)-1)]
        neg = trainList[random.randint(0, len(trainList)-1)]
        if pos[2].startswith('<a>') or pos[3].startswith('<a>') or neg[3].startswith('<a>'):
            #print 'empty string ......'
            continue
        x, m = encode_sent(vocab, pos[2], words_num_dim)
        train_1.append(x)
        mask_1.append(m)
        x, m = encode_sent(vocab, pos[3], words_num_dim)
        train_2.append(x)
        mask_2.append(m)
        x, m = encode_sent(vocab, neg[3], words_num_dim)
        train_3.append(x)
        mask_3.append(m)
        counter += 1
        if counter >= batch_size:
            break
    return np.transpose(np.array(train_1, dtype=config.floatX)), np.transpose(np.array(train_2, dtype=config.floatX)), np.transpose(np.array(train_3, dtype=config.floatX)), np.transpose(np.array(mask_1, dtype=config.floatX)) , np.transpose(np.array(mask_2, dtype=config.floatX)), np.transpose(np.array(mask_3, dtype=config.floatX))
'''


def load_test_data(testList, vocab, index, batch_size, words_num_dim):
    x1, x2, x3, m1, m2, m3 = [], [], [], [], [], []
    for i in range(0, batch_size):
        true_index = index + i
        if true_index >= len(testList):
            true_index = len(testList) - 1
        items = testList[true_index]
        x, m = encode_sent(vocab, items[2], words_num_dim)
        x1.append(x)
        m1.append(m)
        x, m = encode_sent(vocab, items[3], words_num_dim)
        x2.append(x)
        m2.append(m)
        x, m = encode_sent(vocab, items[3], words_num_dim)
        x3.append(x)
        m3.append(m)
    return np.transpose(np.array(x1, dtype=theano.config.floatX)), np.transpose(np.array(x2, dtype=theano.config.floatX)), np.transpose(np.array(x3, dtype=theano.config.floatX)), np.transpose(np.array(m1, dtype=theano.config.floatX)) , np.transpose(np.array(m2, dtype=theano.config.floatX)), np.transpose(np.array(m3, dtype=theano.config.floatX))


def validation(validate_model, testList, vocab, batch_size, words_num_dim):
    index, score_list = int(0), []
    while True:
        x1, x2, x3, m1, m2, m3 = load_test_data(testList, vocab, index, batch_size, words_num_dim)
        batch_scores, nouse = validate_model(x1, x2, x3, m1, m2, m3)
        for score in batch_scores:
            score_list.append(score)
        index += batch_size
        if index >= len(testList):
            break
        #log('Evalution' + str(index), logfile_path)
        print 'Evalution...', str(index)
    sdict, index = {}, int(0)
    qa_count = 0
    for items in testList:
        qid = items[1].split(':')[1]
        if not qid in sdict:
            sdict[qid] = []
        sdict[qid].append((score_list[index], items[0]))
        index += 1
        if int(qid) > qa_count:
            qa_count = int(qid)
    qa_count += 1
    top1 = float(0)
    map_sum = float(0)
    mrr_sum = float(0)
    for qid, items in sdict.items():
        items.sort(key=operator.itemgetter(0), reverse=True)
        #for top1
        score, flag = items[0]
        if flag == '1':
            top1 += 1
        #for mrp
        mrr_index = 0
        for score, flag in items:
            #log('[debug]' + ' qid=' + str(qid) + ' score=' + str(score) + ' flag=' + str(flag), logfilepath)
            mrr_index += 1
            if flag == '1':
                mrr_sum += float(1) / float(mrr_index)

        #for map
        map_index_down = 0
        map_index_up = 0
        temp_map_sum = 0
        for score, flag in items:
            #log('[debug]' + ' qid=' + str(qid) + ' score=' + str(score) + ' flag=' + str(flag))
            map_index_down += 1
            if flag == '1':
                map_index_up += 1
                temp_map_sum += float(map_index_up) / float(map_index_down)
        temp_map_sum /= float(map_index_up)
        map_sum += temp_map_sum
        log('qid = ' + str(qid) + ' / top1 count = ' + str(top1), logfile_path)
    top1 /= float(qa_count)
    mrr_sum /= float(qa_count)
    map_sum /= float(qa_count)
    log('top-1 = ' + str(top1) + ' / ' + 'mrr = ' + str(mrr_sum) + ' / ' + 'map = ' + str(map_sum), logfile_path)


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)


def param_init_cnn(filter_sizes, num_filters, proj_size, tparams, grad_params):
    rng = np.random.RandomState(23455)
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, proj_size)
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:])
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        tparams['cnn_W_' + str(filter_size)] = W
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        b = theano.shared(value=b_values, borrow=True)
        tparams['cnn_b_' + str(filter_size)] = b
        grad_params += [W, b]
    return tparams, grad_params


def param_init_lstm(proj_size, tparams, grad_params):
    W = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    W_t = theano.shared(W, borrow=True)
    tparams[_p('lstm', 'W')] = W_t
    U = np.concatenate([ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size),
                           ortho_weight(proj_size)], axis=1)
    U_t = theano.shared(U, borrow=True)
    tparams[_p('lstm', 'U')] = U_t
    b = np.zeros((4 * proj_size,))
    b_t = theano.shared(b.astype(config.floatX), borrow=True)
    tparams[_p('lstm', 'b')] = b_t
    grad_params += [W_t, U_t, b_t]

    return tparams, grad_params

def dropout_layer(state_before, use_noise, trng):
    proj = T.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj

class LSTM(object):
  def __init__(self, input1, input2, input3, mask1, mask2, mask3, word_embeddings, batch_size, sequence_len, embedding_size, filter_sizes, num_filters, margin_size):
    #proj_size means embedding_size
    #'lstm_W' = [embedding_size, embedding_size]
    #'lstm_U' = [embedding_size, embedding_size]
    #'lstm_b' = [embedding_size]
    proj_size = embedding_size #TODO, what does proj mean
    self.params, tparams = [], {}
    tparams, self.params = param_init_lstm(proj_size, tparams, self.params)
    tparams, self.params = param_init_cnn(filter_sizes, num_filters, proj_size, tparams, self.params)
    lookup_table = theano.shared(word_embeddings, borrow=True)
    tparams['lookup_table'] = lookup_table
    self.params += [lookup_table]

    n_timesteps = input1.shape[0]
    n_samples = input1.shape[1]

    lstm1, lstm_whole1 = self._lstm_net(tparams, input1, sequence_len, batch_size, embedding_size, mask1, proj_size)
    lstm2, lstm_whole2 = self._lstm_net(tparams, input2, sequence_len, batch_size, embedding_size, mask2, proj_size)
    lstm3, lstm_whole3 = self._lstm_net(tparams, input3, sequence_len, batch_size, embedding_size, mask3, proj_size)

    #dimshuffle [sequence_len, batch_size, proj_size] to [batch_size, sequence_len, proj_size]
    cnn_input1 = T.reshape(lstm1.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn_input2 = T.reshape(lstm2.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn_input3 = T.reshape(lstm3.dimshuffle(1, 0, 2), [batch_size, 1, sequence_len, proj_size])
    cnn1 = self._cnn_net(tparams, cnn_input1, batch_size, sequence_len, num_filters, filter_sizes, proj_size)
    cnn2 = self._cnn_net(tparams, cnn_input2, batch_size, sequence_len, num_filters, filter_sizes, proj_size)
    cnn3 = self._cnn_net(tparams, cnn_input3, batch_size, sequence_len, num_filters, filter_sizes, proj_size)

    len1 = T.sqrt(T.sum(cnn1 * cnn1, axis=1))
    len2 = T.sqrt(T.sum(cnn2 * cnn2, axis=1))
    len3 = T.sqrt(T.sum(cnn3 * cnn3, axis=1))

    self.cos12 = T.sum(cnn1 * cnn2, axis=1) / (len1 * len2)
    self.cos13 = T.sum(cnn1 * cnn3, axis=1) / (len1 * len3)

    zero = theano.shared(np.zeros(batch_size, dtype=config.floatX), borrow=True)
    margin = theano.shared(np.full(batch_size, margin_size, dtype=config.floatX), borrow=True)
    diff = T.cast(T.maximum(zero, margin - self.cos12 + self.cos13), dtype=config.floatX)
    self.cost = T.sum(diff, acc_dtype=config.floatX)
    self.accuracy = T.sum(T.cast(T.eq(zero, diff), dtype='int32')) / float(batch_size)

  def _cnn_net(self, tparams, cnn_input, batch_size, sequence_len, num_filters, filter_sizes, proj_size):
    outputs = []
    for filter_size in filter_sizes:
        filter_shape = (num_filters, 1, filter_size, proj_size)
        image_shape = (batch_size, 1, sequence_len, proj_size)
        W = tparams['cnn_W_' + str(filter_size)]
        b = tparams['cnn_b_' + str(filter_size)]
        conv_out = conv2d(input=cnn_input, filters=W, filter_shape=filter_shape, input_shape=image_shape)
        pooled_out = pool.pool_2d(input=conv_out, ds=(sequence_len - filter_size + 1, 1), ignore_border=True, mode='max')
        pooled_active = T.tanh(pooled_out + b.dimshuffle('x', 0, 'x', 'x'))
        outputs.append(pooled_active)
    num_filters_total = num_filters * len(filter_sizes)
    output_tensor = T.reshape(T.concatenate(outputs, axis=1), [batch_size, num_filters_total])
    return output_tensor

  def _lstm_net(self, tparams, _input, sequence_len, batch_size, embedding_size, mask, proj_size):
    input_matrix = tparams['lookup_table'][T.cast(_input.flatten(), dtype="int32")]
    input_x = input_matrix.reshape((sequence_len, batch_size, embedding_size))
    proj, proj_whole = lstm_layer(tparams, input_x, proj_size, prefix='lstm', mask=mask)
    #if useMask == True:
    #proj = (proj * mask[:, :, None]).sum(axis=0)
    #proj = proj / mask.sum(axis=0)[:, None]
    #if options['use_dropout']:
    #proj = dropout_layer(proj, use_noise, trng)
    return proj, proj_whole

#state_below is word_embbeding tensor(3dim)
def lstm_layer(tparams, state_below, proj_size, prefix='lstm', mask=None):
    #dim-0 steps, dim-1 samples(batch_size), dim-3 word_embedding
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #h means hidden output? c means context? so we'll use h?
    #rval[0] = [sequence_len, batch_size, proj_size], rval[1] the same

    #so preact size must equl to x_(lstm input slice)
    #if you want change lstm h(t) size, 'lstm_U' and 'lstm_b'
    #and precat must be changed to another function, like h*U+b
    #see http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    #f(t) = sigmoid(Wf * [h(t-1),x(t)] + bf)
    def _step(m_, x_, h_, c_):
        preact = T.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = T.nnet.sigmoid(_slice(preact, 0, proj_size))
        f = T.nnet.sigmoid(_slice(preact, 1, proj_size))
        o = T.nnet.sigmoid(_slice(preact, 2, proj_size))
        c = T.tanh(_slice(preact, 3, proj_size))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        #if mask(t-1)==0, than make h(t) = h(t-1)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (T.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = proj_size
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              T.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0], rval[1]

def _p(pp, name):
    return '%s_%s' % (pp, name)

def train():
    global logfile_path
    global trainfile
    global train0file
    global test1file

    batch_size = int(256)
    embedding_size = 300
    words_num_dim = 50
    learning_rate = 0.005
    margin_size = 0.05
    n_epochs = 20000
    validation_freq = 10
    filter_sizes = [1, 2, 3, 5]
    num_filters = 500

    logfile_path = os.path.join(logfile_path, 'LSTM-' + GetNowTime() + '-' \
                   + 'batch_size-' + str(batch_size) + '-' \
                   + 'num_filters-' + str(num_filters) + '-' \
                   + 'embedding_size-' + str(embedding_size) + '-' \
                   + 'n_epochs-' + str(n_epochs) + '-' \
                   + 'freq-' + str(validation_freq) + '-' \
                   + '-log.txt')

    log("New start ...", logfile_path)
    log(str(time.asctime(time.localtime(time.time()))), logfile_path)
    log("batch_size = " + str(batch_size), logfile_path)
    log("filter_sizes = " + str(filter_sizes), logfile_path)
    log("num_filters = " + str(num_filters), logfile_path)
    log("embedding_size = " + str(embedding_size), logfile_path)
    log("margin_size = " + str(margin_size), logfile_path)
    log("learning_rate = " + str(learning_rate), logfile_path)
    log("n_epochs = " + str(n_epochs), logfile_path)
    log("words_num_dim = " + str(words_num_dim), logfile_path)
    log("validation_freq = " + str(validation_freq), logfile_path)
    log("train_1_file = " + str(trainfile.split('/')[-1]), logfile_path)
    log("train_0_file = " + str(train0file.split('/')[-1]), logfile_path)
    log("test_file = " + str(test1file.split('/')[-1]), logfile_path)
    log("vector_file = " + str(vectorsfile.split('/')[-1]), logfile_path)

    vocab = build_vocab()
    word_embeddings = load_word_embeddings(vocab, embedding_size)
    trainList = load_train_list()
    testList = load_test_list()
    train0Dict = load_train0_dict()
    train_x1, train_x2, train_x3, mask1, mask2, mask3 = load_train_data_from_2files(train0Dict, trainList, vocab, batch_size, words_num_dim)
    x1, x2, x3 = T.fmatrix('x1'), T.fmatrix('x2'), T.fmatrix('x3')
    m1, m2, m3 = T.fmatrix('m1'), T.fmatrix('m2'), T.fmatrix('m3')
    model = LSTM(
        input1=x1, input2=x2, input3=x3,
        mask1=m1, mask2=m2, mask3=m3,
        word_embeddings=word_embeddings,
        batch_size=batch_size,
        sequence_len=train_x1.shape[0], #row is sequence_len
        embedding_size=embedding_size,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        margin_size = margin_size)

    cost, cos12, cos13 = model.cost, model.cos12, model.cos13
    params, accuracy = model.params, model.accuracy
    grads = T.grad(cost, params)
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    p1, p2, p3 = T.fmatrix('p1'), T.fmatrix('p2'), T.fmatrix('p3')
    q1, q2, q3 = T.fmatrix('q1'), T.fmatrix('q2'), T.fmatrix('q3')
    train_model = theano.function(
        [p1, p2, p3, q1, q2, q3], 
        [cost, accuracy], 
        updates=updates,
        givens={
            x1: p1, x2: p2, x3: p3, m1: q1, m2: q2, m3: q3
        }
    )

    v1, v2, v3 = T.matrix('v1'), T.matrix('v2'), T.matrix('v3')
    u1, u2, u3 = T.matrix('u1'), T.matrix('u2'), T.matrix('u3')
    validate_model = theano.function(
        inputs=[v1, v2, v3, u1, u2, u3],
        outputs=[cos12, cos13],
        #updates=updates,
        givens={
            x1: v1, x2: v2, x3: v3, m1: u1, m2: u2, m3: u3
        }
    )

    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_x1, train_x2, train_x3, mask1, mask2, mask3 = load_train_data_from_2files(train0Dict, trainList, vocab, batch_size, words_num_dim)
        #print('train_x1, train_x2, train_x3')
        #print(train_x1.shape, train_x2.shape, train_x3.shape)
        cost_ij, acc = train_model(train_x1, train_x2, train_x3, mask1, mask2, mask3)
        log('load data done ...... epoch:' + str(epoch) + ' cost:' + str(cost_ij) + ', acc:' + str(acc), logfile_path)
        if epoch % validation_freq == 0:
            log('Evaluation ......', logfile_path)
            validation(validate_model, testList, vocab, batch_size, words_num_dim)

if __name__ == '__main__':
    train()
