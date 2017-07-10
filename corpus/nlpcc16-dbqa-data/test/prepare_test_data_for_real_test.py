#!/usr/bin/python
#  -*- coding: utf-8 -*-

import os
import codecs
import sys

sys.path.append('/home/zhengxing/QAModel')
from util.word_segment import *
from util.LCS import *
from util.normalWordList import *


#reload(sys)
#sys.setdefaultencoding('utf-8')


if __name__ == '__main__':
    filepath = os.path.abspath('.')
    filename = 'test'
    version_num = '1'
    test_filename = filename + '_data_version_' + version_num
    stop_word_file_path = '../../lib/chStopWordsSimple.txt'
    segment_words_number = 200

    fopw = codecs.open(test_filename, 'w', encoding='utf-8')

    lineNo = 0
    qid = 0
    dic = {}
    for line in codecs.open(os.path.join(filepath, filename), 'r', encoding='utf-8'):
        words = line.strip().split('\t')
        if len(words) == 3:
            lineNo += 1
            print lineNo
            if words[0] not in dic:
                dic[words[0]] = qid
                qid += 1
            currentId = dic[words[0]]
            str_a_list = get_segment_word_list(words[0], stop_word_file_path)
            str_b_list = get_segment_word_list(words[1], stop_word_file_path)
            lcs_a_b_list = least_common_string(str_a_list, str_b_list)
            #str_a_list.extend(lcs_a_b_list)
            str_b_list.extend(lcs_a_b_list)
            str_a = normal_word_list(str_a_list, segment_words_number)
            str_b = normal_word_list(str_b_list, segment_words_number)
            fopw.write(words[2] + '\t'
                       + 'qid:' + str(currentId) + '\t'
                       + str_a
                       + '\t' + str_b
                       + '\n')
    fopw.close()