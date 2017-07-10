#!/usr/bin/python
#  -*- coding: utf-8 -*-
__author__ = 'xing'

import os
import sys

sys.path.append('/home/zhengxing/QAModel')

import codecs
import json
from util.word_segment import *
from util.normalWordList import *
from util.LCS import *
from util.crawlUrl import *


#data is json's format
def get_data_from_line(data):
    json_data = json.loads(data)
    return json_data

def solve_title(title):
    title = title.strip().replace(' ', ',').replace('\t', ',')
    return title

def get_title(filepath):
    t_dict = {}
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line != '' and line is not None:
                items = line.split('&&&&&')
                if len(items) > 0:
                    url = items[0]
                    title = ''
                    if len(items) == 2:
                        title = items[1]
                    if url not in t_dict:
                        t_dict[url] = title
    return t_dict


'''
def get_title_from_url(url, t_dict):
    title = ""
    if url in t_dict:
        title = t_dict[url]
    if title == "":
        title = str(url)
    return title
'''

def get_title_from_url(url, t_dict):
    title = ""
    if url in t_dict:
        title = t_dict[url]
    return title


if __name__ == '__main__':
    version_num = 'all-2-10-just-url-pure'
    filepath = '.'
    filename = 'json.longshort.new.all'
    filename1 = filename + '_train_part_1_data_version_' + version_num
    filename0 = filename + '_train_part_0_data_version_' + version_num
    title_filepath = 'json.longshort.new.all.new'
    stop_word_file_path = '../../lib/chStopWordsSimple.txt'
    segment_words_number = 100

    t_dict = get_title(os.path.join(filepath, title_filepath))

    fw1 = codecs.open(os.path.join(filepath, filename1), 'w', encoding='utf-8')
    fw0 = codecs.open(os.path.join(filepath, filename0), 'w', encoding='utf-8')

    #count the number of lines
    with open(os.path.join(filepath, filename), 'r') as fr:
        line_no = 0
        for line in fr:
            line_no += 1

    query_max_words_count = float(0)
    query_avg_words_count = float(0)
    answer_max_words_count = float(0)
    answer_avg_words_count = float(0)
    #generate train data
    #dont use codecs.open
    print 'start train data...'
    with open(os.path.join(filepath, filename), 'r') as fr:
        line_no = 0
        for line in fr:
            line_no += 1
            line = str(line.strip())
            if line != '':
                dict_data = get_data_from_line(line)
                query_id = str(dict_data['query_id'])
                query = str(dict_data['query'])
                passages = dict_data['passages']
                print 'generate train...' + query_id + '...'
                for passage in passages:
                    dict_passage = passage
                    passage_id = str(dict_passage['passage_id'])
                    passage_url = str(dict_passage['url'])
                    passage_text = str(dict_passage['passage_text'])
                    passage_label = str(dict_passage['label'])
                    #get title from url
                    passage_title = get_title_from_url(passage_url, t_dict)
                    if passage_title == "":
                        passage_title = passage_text
                    alist = get_filtered_segment_word_list(query, stop_word_file_path)
                    blist = get_filtered_segment_word_list(passage_title, stop_word_file_path)
                    #lcs list
                    lcs_list = least_common_string(alist, blist)
                    temp_list = blist
                    blist = lcs_list
                    blist.extend(temp_list)
                    query_max_words_count = max(query_max_words_count, len(alist))
                    answer_max_words_count = max(answer_max_words_count, len(blist))
                    query_avg_words_count += len(alist)
                    answer_avg_words_count += len(blist)
                    str_a = normal_word_list(alist, segment_words_number)
                    str_b = normal_word_list(blist, segment_words_number)
                    #seperate the right & wrong answer
                    if passage_label == '2':
                        fw1.write(query_id + '_1' + '\t'
                                  + 'qid:' + query_id + '\t'
                                  + str_a + '\t'
                                  + str_b
                                  + '\n')
                    else:
                        fw0.write(query_id + '_0' + '\t'
                                  + 'qid:' + query_id + '\t'
                                  + str_a + '\t'
                                  + str_b
                                  + '\n')
                print query_id + '......'
    print 'max query = ', query_max_words_count, ' avg query = ', query_avg_words_count / line_no
    print 'max answer = ', answer_max_words_count, ' avg answer = ', answer_avg_words_count / line_no
    fw0.close()
    fw1.close()
