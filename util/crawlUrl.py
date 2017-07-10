#coding=utf-8

import sys
from bs4 import BeautifulSoup as bs
import urllib2 as url
import codecs, chardet

def get_html_content_from_url_link(url_link):
    try:
        res = url.urlopen(url_link, timeout=10)
        return res.read()
    except:
        return None
        #print 'ex', url_link

def get_title_from_url_link(url_link):
    content = get_html_content_from_url_link(url_link)
    if content is None:
        return ""
    p1 = content.find('<title>')
    p2 = content.find('</title>')
    if p1 < 0 or p2 < p1:
        return ""
    return content[p1 + 7: p2]

if __name__ == '__main__':
    #print get_title_from_url_link('')
    #print get_title_from_url_link('http://zhidao.baidu.com/question/46630494')
    t1 = get_title_from_url_link('http://zhidao.baidu.com/question/46630494')
    print chardet.detect(t1), t1
