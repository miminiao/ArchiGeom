from lib.geom import *
from copy import copy

line1=LineSeg(Node(0,0),Node(1,1))
line1.lw=1234
line1.rw=12344
line2=copy(line1)

line1.s=Node(2,2)
print(line1,line2)