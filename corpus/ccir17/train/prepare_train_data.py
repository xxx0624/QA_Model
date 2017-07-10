#!/usr/bin/python
#  -*- coding: utf-8 -*-
__author__ = 'xing'

import os
import sys

sys.path.append('/home/zhengxing/QAModel')

import codecs
import json
from util import segment_word_filter_pos


#data is json's format
def get_data_from_line(data):
    json_data = json.loads(data)
    return json_data


if __name__ == '__main__':
    filepath = '.'
    filename = 'train.1.json'
    version_num = '1'
    filename1 = filename + '1_data_version_' + version_num
    filename0 = filename + '0_data_version_' + version_num
    stop_word_file_path = '../../lib/chStopWordsSimple.txt'
    segment_words_number = 200

    fw1 = codecs.open(os.path.join(filepath, filename1), 'w', encoding='utf-8')
    fw0 = codecs.open(os.path.join(filepath, filename0), 'w', encoding='utf-8')
    #dont use codecs.open
    with open(os.path.join(filepath, filename), 'r') as fr:
        line_no = 0
        for line in fr:
            line_no += 1
            line = str(line.strip())
            #print type(line)
            #print str(line)
            if line != '':
                dict_data = get_data_from_line(line)
                query_id = str(dict_data['query_id'])
                query = str(dict_data['query'])
                passages = dict_data['passages']
                print query_id + '...'
                for passage in passages:
                    #print type(passage)
                    #print str(passage)
                    #dict_passage = get_data_from_line(str(passage))
                    dict_passage = passage
                    passage_id = str(dict_passage['passage_id'])
                    passage_url = str(dict_passage['url'])
                    passage_text = str(dict_passage['passage_text'])
                    passage_label = str(dict_passage['label'])
                    #seperate the right & wrong answer
                    if passage_label == '2':
                        fw1.write(query_id + '_1' + '\t'
                                  + 'qid:' + query_id + '\t'
                                  + segment_word_filter_pos(query, segment_words_number, stop_word_file_path) + '\t'
                                  + segment_word_filter_pos(passage_text, segment_words_number, stop_word_file_path)
                                  + '\n')
                    else:
                        fw0.write(query_id + '_0' + '\t'
                                  + 'qid:' + query_id + '\t'
                                  + segment_word_filter_pos(query, segment_words_number, stop_word_file_path) + '\t'
                                  + segment_word_filter_pos(passage_text, segment_words_number, stop_word_file_path)
                                  + '\n')
                print query_id + '......'
    fw0.close()
    fw1.close()
