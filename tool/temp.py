from lib.geom import Node
from lib.index import KDTree,_KDTreeNode
if __name__=="__main__":
    import random
    n=100
    limits=1000
    nodes=[Node(random.random()*limits,random.random()*limits) for i in range(n)]
    kdtree=KDTree(nodes)
    
    import matplotlib.pyplot as plt
    plt.scatter([node.x for node in nodes],[node.y for node in nodes])

    plt.show()