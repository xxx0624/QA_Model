__author__ = 'xing'
import os, sys

def least_common_string(str_a, str_b):
    lena = len(str_a)
    lenb = len(str_b)
    #init
    dp = []
    for i in range(lena + 1):
        dp.append([])
        for j in range(lenb + 1):
            dp[i].append(0)
    #dp
    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    #get the same string list
    same_str = []
    index_a = lena
    index_b = lenb
    while index_a >= 1 and index_b >= 1:
        if str_a[index_a - 1] == str_b[index_b - 1]:
            same_str.append(str_a[index_a - 1])
            #print str_a[index_a - 1]
            index_a -= 1
            index_b -= 1
        elif dp[index_a][index_b - 1] > dp[index_a - 1][index_b]:
            index_b -= 1
        else:
            index_a -= 1
    same_str.reverse()
    return same_str


'''
#test
a=[1,3,5,7,9]
b=[2,3,5,6,6,6,9]
c=least_common_string(a,b)
print c
'''