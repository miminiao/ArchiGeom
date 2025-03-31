from lib.utils import ListTool

import random
random.seed(2)
n=1000
# a=[random.randint(0,1000) for _ in range(n)]
a=list(range(n))
random.shuffle(a)
c=a[:]
ListTool.qsort(c)
b=sorted(a)
assert b==c
# a.reverse()
# for k in range(n):
#     idx=ListTool.get_nth(a,k)
#     assert a[idx]==k
