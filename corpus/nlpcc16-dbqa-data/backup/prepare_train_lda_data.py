#!/usr/bin/python
#  -*- coding: utf-8 -*-

import os
import sys

sys.path.append('/home/zhengxing/QAModel')
import codecs

from util.word_segment import *
from util.LCS import *
from util.normalWordList import *


#reload(sys)
#sys.setdefaultencoding('utf-8')

if __name__ == '__main__':
    filepath = os.path.abspath('.')
    filename = 'train'
    version_num = 'only-answer-lcs-lda'
    filename1 = filename + '1_data_version_' + version_num
    filename0 = filename + '0_data_version_' + version_num
    stop_word_file_path = '../../lib/chStopWordsSimple.txt'
    segment_words_number = 100

    fopw1 = codecs.open(os.path.join(filepath, filename1), 'w', encoding='utf-8')
    fopw0 = codecs.open(os.path.join(filepath, filename0), 'w', encoding='utf-8')

    lineNo = 0
    qid = 0
    dic = {}
    max_length = 0
    for line in codecs.open(os.path.join(filepath, filename), 'r', encoding='utf-8'):
        words = line.strip().split('\t')
        if len(words) == 3:
            lineNo += 1
            print lineNo
            if words[0] not in dic:
                dic[words[0]] = qid
                qid += 1
            currentId = dic[words[0]]
            #str_a = segment_word_filter_pos(words[0], segment_words_number, stop_word_file_path)
            #str_b = segment_word_filter_pos(words[1], segment_words_number, stop_word_file_path)
            str_a_list = get_segment_word_list(words[0], stop_word_file_path)
            str_b_list = get_segment_word_list(words[1], stop_word_file_path)
            lcs_a_b_list = least_common_string(str_a_list, str_b_list)
            #str_a_list.extend(lcs_a_b_list)
            str_b_list.extend(lcs_a_b_list)
            str_b_list.append(str(lineNo) + 'TrainLineNo')
            str_a = normal_word_list(str_a_list, segment_words_number)
            str_b = normal_word_list(str_b_list, segment_words_number)
            print 'Length = ', len(str_a_list), len(str_b_list)
            max_length = max(max_length, max(len(str_a_list), len(str_b_list)))
            if words[2] == '1':
                fopw1.write(str(currentId) + '_1' + '\t'
                            + 'qid:' + str(currentId) + '\t'
                            + str_a + '\t'
                            + str_b
                            + '\n')
            if words[2] == '0':
                fopw0.write(str(currentId) + '_0' + '\t'
                            + 'qid:' + str(currentId) + '\t'
                            + str_a
                            + '\t' + str_b
                            + '\n')
    print 'max length = ', max_length
    fopw0.close()
    fopw1.close()

