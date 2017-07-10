__author__ = 'xing'
import time

def GetNowTime():
    return str(time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time())))
