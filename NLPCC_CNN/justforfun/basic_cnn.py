#!/usr/bin/python
#  -*- coding: utf-8 -*-

import codecs,os,sys
import numpy as np
#from gensim.models import Word2Vec
#from gensim.models import KeyedVectors
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d


theano.config.floatX = 'float32'

'''
rng = np.random.RandomState(23455)
input = T.tensor4(name = 'input')
w_shape = (1, 1, 2, 2)
fan_in = np.prod(w_shape[1:])
fan_out = w_shape[0] * np.prod(w_shape[2:])
w_bound = np.sqrt(6. / (fan_in + fan_out))
W = theano.shared(
        np.asarray(
            rng.uniform(low = -1.0 / w_bound, high = 1.0 / w_bound, size = w_shape),
            dtype = theano.config.floatX
        ),
        name = 'W'
    )
b_shape = (2,)
b = theano.shared(
        np.asarray(
            rng.uniform(low = -.5, high = .5, size = b_shape),
            dtype = theano.config.floatX
        ),
        name = 'b'
    )
conv_out = conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f_cnn = theano.function([input], output)

'''
alist = [[1,2,3,4,5,6,7,8,9,10]]
width = len(alist)
height = 1
alist = np.asarray(alist, dtype=theano.config.floatX) / 10.0
alist_rgb = alist.swapaxes(0,2).swapaxes(1,2).reshape(1,1,height,width) #(1,height,width)

