__author__ = 'xing'

class Query:
    def __init__(self):
        self.query_id = ''
        self.query = ''
        self.passages = []
    def __str__(self):
        ans = '{' +\
                '\"query_id\":' + '\"' + self.query_id + '\"' +',' +\
                '\"query\":' + '\"' + self.query + '\"' +',' +\
                '\"passages\":['
        flag = 0
        for passage in self.passages:
            if flag == 0:
                ans += passage.__str__()
                flag += 1
            else:
                ans += ',' + passage.__str__()
        ans += ']' + '}'
        return ans

