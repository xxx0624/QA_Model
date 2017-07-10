__author__ = 'xing'

'''
THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1'  python a
'''

a = [1,3,2]
sorted(a, reverse=True)
print a

b = 'a123,dfasdf,_<a>_sfasdf_<a>_asdfa_<a>_<a>'
print b.strip('_<a>')

c = [(1,2,31),(4,5,61)]
def co(x, y):
    if x[2] < y[2]:
        return 1
    else:
        return -1
c.sort(cmp=co)
print c

e = [(1,2),(0,5)]
def co2(x, y):
    if x[1] < y[1]:
        return 1
    return -1
e.sort(cmp=co2)
print e

d = [1,4,2]
d.sort(reverse=True)
print d

f = {'1':3, '5':6}
if '5' in f:
    print '233'

g = [1,3,5,7]
for gg in g:
    gg = 4
print g

h = [[] for n in range(5)]
print len(h), h

for i in range(2,3):
    print 'wa...', i

for i in range(9,3,-1):
    print i

