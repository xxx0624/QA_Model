__author__ = 'xing'
import os, sys

def normal_word_list(word_list, words_len):
    count = 0
    res = ''
    for i in word_list:
        if res == '':
            res += i
            count += 1
            if count >= words_len:
                break
        else:
            res += '_' + i
            count += 1
            if count >= words_len:
                break
    if count < words_len:
        for i in range(count, words_len):
            if res == '':
                res += '<a>'
            else:
                res += '_<a>'
    return res