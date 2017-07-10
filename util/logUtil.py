__author__ = 'xing'
from timeUtil import GetNowTime

def log(info, logpath):
    print info
    with open(logpath, "a+") as f:
        f.write('[' + GetNowTime() +'] ' + info)
        f.write("\n")

def new_log(info, logpath):
    print info
    with open(logpath, 'w') as f:
        f.write('[' + GetNowTime() + ']' + info)
        f.write('\n')
