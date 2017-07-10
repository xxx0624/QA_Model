#!/usr/bin/python
#  -*- coding: utf-8 -*-
__author__ = 'xing'

import os
import sys

sys.path.append('/home/zhengxing/QAModel')
#reload(sys)
#sys.setdefaultencoding('utf-8')

import codecs, chardet
import json
from util.word_segment import *
from util.normalWordList import *
from util.LCS import *
from util.crawlUrl import *
from util.ccir_data.Passage import *
from util.ccir_data.Query import *

#data is json's format
def get_data_from_line(data):
    json_data = json.loads(data)
    return json_data


'''
generate new raw data with the urls
new raw data is url and the title
'''


if __name__ == '__main__':
    filepath = '.'
    filename = 'json.testset'
    new_filename = filename + '.new'

    separate_tag = '&&&&&'

    url_dict = {'http://1882499479452806828.qganjue.com/':'',
                'http://1703093482293933980.qganjue.com/':'',
                'http://753496053245957964.qganjue.com/':'',
                'http://136139761381299765.qganjue.com/':'',
                'http://342528097.qganjue.com/':'',
                'http://570034005.qganjue.com/':'',
                'http://bbs.zol.com.cn/quanzi/d643_899899.html':'',
                'http://2137918422511198948.qganjue.com/':'',
                'http://176977101772888644.qganjue.com/':'',
                'http://www.newsmth.net/nForum/article/Microwave_RFTech/549':''
                }
    if os.path.exists(os.path.join(filepath, new_filename)):
        with codecs.open(os.path.join(filepath, new_filename), 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(separate_tag)
                if line != None and line != '' and len(line) == 2:
                    url_dict[line[0]] = line[1]

    fw = codecs.open(os.path.join(filepath, new_filename), 'a+', encoding='utf-8', errors="ignore")

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
                #print chardet.detect(query_id)
                query = str(dict_data['query'])
                passages = dict_data['passages']
                print query_id + '...'
                passage_list = {'passages':[]}
                query_class = Query()
                query_class.query = query
                query_class.query_id = query_id
                query_class.passages = []
                for passage in passages:
                    dict_passage = passage
                    passage_id = str(dict_passage['passage_id'])
                    passage_url = str(dict_passage['url'])
                    print passage_url
                    if not passage_url in url_dict and not 'qganjue.com/' in passage_url:
                        passage_title = get_title_from_url_link(passage_url)
                        passage_title = passage_title.strip().replace('\t', '').replace('\n', '')
                        #print chardet.detect(passage_title)
                        print passage_title
                        #detect encoding
                        if chardet.detect(passage_title)['encoding'] is None:
                            passage_title = ""
                        elif chardet.detect(passage_title)['encoding'] == 'TIS-620':
                            passage_title = ""
                        elif chardet.detect(passage_title)['encoding'] == 'GB2312':
                            passage_title = passage_title.decode('GBK').encode('utf-8')
                        elif chardet.detect(passage_title)['encoding'] == 'KOI8-R':
                            passage_title = passage_title.decode('GBK').encode('utf-8')
                        elif chardet.detect(passage_title)['encoding'] == 'utf-8':
                            passage_title = passage_title.decode('utf-8').encode('utf-8')
                        #elif 'ISO-8859' in chardet.detect(passage_title)['encoding']:
                        #    passage_title = passage_title.decode(chardet.detect(passage_title)['encoding']).encode('utf-8')
                        else:
                            #IBM855
                            passage_title = ""
                        url_dict[passage_url] = passage_title
                        fw.write(passage_url + separate_tag + passage_title + '\n')
                        fw.flush()
    fw.close()
