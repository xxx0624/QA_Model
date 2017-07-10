#!/usr/bin/python
#  -*- coding: utf-8 -*-

__author__ = 'xing'

import os,sys,codecs,math
sys.path.append('/home/zhengxing/QAModel')
from util.word_segment import *

def comp(x, y):
    if x[1] < y[1]:
        return 1
    else:
        return -1


if __name__ == '__main__':
    filepath = '.'
    filename = 'train'
    single_word_idf_file = filename + '.word-idf'
    top_k = 3
    expand_with_topK_file = filename + '.expand-' + str(top_k) + 'idf'

    all_answer = []
    answer_words_dict = {}
    with codecs.open(os.path.join(filepath, filename), 'r', encoding='utf-8') as fr:
        index = 0
        for line in fr:
            print 'collect answer...', index
            index += 1
            line = line.strip()
            answer = line.split('\t')[1].strip()
            answer_words = get_filtered_segment_word_list(answer, None)
            all_answer.append(answer_words)

    all_answer_count = len(all_answer)
    index = 0
    fw = codecs.open(os.path.join(filepath, single_word_idf_file), 'w', encoding='utf-8')
    for answer_words in all_answer:
        print 'calculate idf...', index
        index += 1
        #start calculate
        for w in answer_words:
            if w not in answer_words_dict:
                idf_count = float(1)
                for temp_answer in all_answer:
                    if w in temp_answer:
                        idf_count += float(1)
                answer_words_dict[w] = math.log(all_answer_count, 10) / math.log(idf_count, 10)
                fw.write(w + ',' + str(answer_words_dict[w]) + '\n')
    fw.close()

    answer_words_idf_list = [[] for i in range(all_answer_count)]
    for index in range(all_answer_count):
        print 'sort...', index
        for w in all_answer[index]:
            answer_words_idf_list[index].append((w, answer_words_dict[w]))
        answer_words_idf_list[index].sort(cmp=comp)

    #generate new file with topK word
    fw = codecs.open(os.path.join(filepath, expand_with_topK_file), 'w', encoding='utf-8')
    fw_bk = codecs.open(os.path.join(filepath, expand_with_topK_file + '.NotUsed'), 'w', encoding='utf-8')
    with codecs.open(os.path.join(filepath, filename), 'r', encoding='utf-8') as fr:
        index = 0
        for line in fr:
            print 'generate...', index
            line = line.strip()
            answer = line.split('\t')[1].strip()
            query = line.split('\t')[0].strip()
            label = line.split('\t')[2].strip()
            temp_bk = ''
            for i in range(len(answer_words_idf_list[index])):
                if i < top_k:
                    answer += '。' + answer_words_idf_list[index][i][0]
                temp_bk += answer_words_idf_list[index][i][0] + ',' + str(answer_words_idf_list[index][i][1]) + '/'
            fw.write(query + '\t' + answer + '\t' + label + '\n')
            fw_bk.write(temp_bk + '\n')
            index += 1
    fw.close()
    fw_bk.close()
