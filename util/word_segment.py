#!/usr/bin/python
#  -*- coding: utf-8 -*-
import os, sys
import jieba
import jieba.analyse
import jieba.posseg as pseg

reload(sys)
sys.setdefaultencoding('utf-8')

#this is used for version1's train & test
def segment_word(sentence, normal_word_list_size, stop_words_path):
    if stop_words_path is None:
        pass
    else:
        jieba.analyse.set_stop_words(stop_words_path)
    words = jieba.cut(sentence, cut_all=False)
    word_list = ''
    count = 0
    for w in words:
        if count < normal_word_list_size:
            count += 1
            if word_list == '':
                word_list = w
            else:
                word_list += '_' + w
    if count < normal_word_list_size:
        for i in range(count, normal_word_list_size):
            if word_list == '':
                word_list = '<a>'
            else:
                word_list += '_<a>'
    return word_list

#this is used for version2's train & test
def judge_word_flag2(flag):
    if flag[0] == 'r' \
        or flag[0] == 'm' \
        or flag[0] == 'q' \
        or flag[0] == 'p' \
        or flag[0] == 'c' \
        or flag[0] == 'u' \
        or flag[0] == 'e' \
        or flag[0] == 'y' \
        or flag[0] == 'o' \
        or flag[0] == 'w' \
        or flag[0] == 'x':
        return True
    return False


#this is used for version3's train & test
def judge_word_flag3(flag):
    if flag[0] == 'c' \
        or flag[0] == 'u' \
        or flag[0] == 'e' \
        or flag[0] == 'y' \
        or flag[0] == 'o' \
        or flag[0] == 'w' \
        or flag[0] == 'x':
        return True
    return False

def segment_word_filter_pos(sentence, normal_word_list_size, stop_words_path):
    if stop_words_path is None:
        pass
    else:
        jieba.analyse.set_stop_words(stop_words_path)
    words = pseg.cut(sentence)
    word_list = ''
    count = 0
    for w in words:
        word = w.word
        flag = str(w.flag)
        if judge_word_flag3(flag) is True:
            continue
        if count < normal_word_list_size:
            count += 1
            if word_list == '':
                word_list = word
            else:
                word_list += '_' + word
    if count < normal_word_list_size:
        for i in range(count, normal_word_list_size):
            if word_list == '':
                word_list = '<a>'
            else:
                word_list += '_<a>'
    return word_list


def get_filtered_segment_word_list(sentence, stop_words_path):
    if stop_words_path is None:
        pass
    else:
        jieba.analyse.set_stop_words(stop_words_path)
    words = pseg.cut(sentence)
    word_list = []
    for w in words:
        flag = str(w.flag)
        if judge_word_flag3(flag) is True:
            continue
        word_list.append(w.word)
    return word_list


def get_segment_word_list(sentence, stop_words_path):
    if stop_words_path is None:
        pass
    else:
        jieba.analyse.set_stop_words(stop_words_path)
    words = pseg.cut(sentence)
    word_list = []
    for w in words:
        word_list.append(w.word)
    return word_list


if __name__ == '__main__':
    s = "今天'我[最喜欢]中国，天安门。广场,它是\"一只.猫"
    slist = get_filtered_segment_word_list(s, None)
    for w in slist:
        print w