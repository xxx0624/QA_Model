__author__ = 'xing'
import jieba.posseg as pseg
import codecs

with codecs.open('training.csv', 'r', encoding='utf-8') as fr:
    fw = codecs.open('training.seg.csv', 'w', encoding='utf-8')
    line_cnt = 0
    word_cnt = 0
    for line in fr:
        line = line.strip()
        if line != "":
            label = line.split(",")[0]
            line = line.split(",")[1]
            words = pseg.cut(line)
            line_cnt += 1
            fw.write(label + ',')
            id = 0
            for w in words:
                # print w.word, w.flag
                if id == 0:
                    fw.write(w.word + '/' + w.flag)
                else:
                    fw.write(' ' + w.word + '/' + w.flag)
                id += 1
            word_cnt += id
            fw.write('\n')
    fw.close()
    print 'avg = ', word_cnt / line_cnt

with codecs.open('testing.csv', 'r', encoding='utf-8') as fr:
    fw = codecs.open('testing.seg.csv', 'w', encoding='utf-8')
    word_cnt = 0
    line_cnt = 0
    for line in fr:
        line = line.strip()
        if line != "":
            label = line.split(",")[0]
            line = line.split(",")[1]
            words = pseg.cut(line)
            line_cnt += 1
            fw.write(label + ',')
            id = 0
            for w in words:
                # print w.word, w.flag
                if id == 0:
                    fw.write(w.word + '/' + w.flag)
                else:
                    fw.write(' ' + w.word + '/' + w.flag)
                id += 1
            word_cnt += id
            fw.write('\n')
    fw.close()
    print 'avg = ', word_cnt / line_cnt
