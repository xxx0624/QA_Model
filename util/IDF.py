__author__ = 'xing'
import math

def comp(x, y):
    if x[1] < y[1]:
        return 1
    return -1

def get_idf(passage_list):
    '''
    :param passage_list:
        [
            [w1,w2,w3...],
            [w2,w4,w5...],
            ...
        ]
    :return:
    '''
    passage_count = float(len(passage_list))
    idf_passage_count = {}
    idf_list = []
    for index in range(len(passage_list)):
        idf_list.append([])
    for index in range(len(passage_list)):
        for w in passage_list[index]:
            if w not in idf_passage_count:
                count = float(0)
                for passage in passage_list:
                    for tw in passage:
                        if tw in w or w in tw:
                            count += 1
                            break
                idf_passage_count[w] = math.log(passage_count, 10) / math.log(count, 10)
            idf_list[index].append((w, idf_passage_count[w]))
        idf_list[index].sort(cmp=comp)
    return idf_list
