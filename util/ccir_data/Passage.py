__author__ = 'xing'

class Passage:
    def __init__(self):
        self.url = ''
        self.passage_id = ''
        self.passage_text = ''
        self.passage_body = ''
        self.label = ''
    def __str__(self):
        return '{'+\
               '\"url\":' + '\"' +self.url + '\"' +',' + \
               '\"passage_id\":' +'\"' + self.passage_id +'\"' + ',' +\
               '\"passage_text\":' +'\"' + self.passage_text + '\"' +',' + \
               '\"passage_body\":' + '\"' +self.passage_body +'\"' + ',' + \
               '\"label\":' + '\"' +self.label +'\"' + \
               +'}'