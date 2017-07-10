__author__ = 'xing'
import os,sys,codecs

sys.path.append('/home/zhengxing/QAModel')

from util.word_segment import *

if __name__ == '__main__':
    filepath = '.'
    filename = 'train'

    new_filename = filename + '-lda-data'

    fw = codecs.open(os.path.join(filepath, new_filename), 'w', encoding='utf-8')

    with codecs.open(os.path.join(filepath, filename), 'r', encoding='utf-8') as f:
        line_no = 0
        for line in f:
            line_no += 1
            print line_no, '...'
            line = line.strip()
            if line is not None and line != '':
                items = line.split('\t')
                if len(items) == 3:
                    question = items[0]
                    answer = items[1]
                    label = items[2]
                    word_list = get_filtered_segment_word_list(answer, None)
                    words_record = ''
                    for w in word_list:
                        if words_record == '':
                            words_record = w
                        else:
                            words_record += ' ' + w
                    if words_record != '':
                        words_record += '\n'
                        fw.write(words_record)
    fw.close()