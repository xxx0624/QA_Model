#coding=utf-8
__author__ = 'xing'
sentence = "a b c d"
print sentence.split()

from util.word_segment import *
sentence = u'6/7=0.857142857的6/7=0.857142857是什么'
print get_segment_word_list(sentence, None)