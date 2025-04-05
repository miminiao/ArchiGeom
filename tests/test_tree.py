"""测试查找树相关操作"""

import pytest
from lib.geom import Node,Box,Circle
from lib.index import BSTree,_BSTreeNode,AVLTree,SegmentTree,KDTree,_KDTreeNode
from tests.utils import read_case,write_stdout
from lib.interval import Interval1d
import random

ROOT="./tests/tree/"

SEG_TREE=(ROOT+"seg_tree/",5)

input_num=1000
def test_bst_insert():
    """测试bst插入"""
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    bst=BSTree(inputs)
    res=bst.traverse()
    std=sorted(inputs)
    assert res==std
    for node in bst.traverse_nodes():
        assert node.get_root()==bst.root    
def test_bst_find():
    """测试bst查找"""
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    bst=BSTree(inputs)
    for i in range(input_num):
        node=bst.find(inputs[i])
        assert node is not None
    assert bst.find(-1) is None
def test_bst_remove():
    """测试bst删除"""
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    bst=BSTree(inputs)
    random.shuffle(inputs)
    for i in range(input_num):
        bst.remove(inputs[i])
        res=bst.traverse()
        std=sorted(inputs[i+1:])
        assert res==std
def test_l_rotete():
    """测试bst左旋"""
    inputs=[2,1,4,3,5]
    avl=BSTree(inputs)
    avl.root=avl.root._l_rotate()
    assert avl.root.obj==4 and avl.traverse()==[1,2,3,4,5]
def test_r_rotete():
    """测试bst右旋"""
    inputs=[4,5,2,3,1]
    avl=BSTree(inputs)
    avl.root=avl.root._r_rotate()
    assert avl.root.obj==2 and avl.traverse()==[1,2,3,4,5]    
def test_avl_insert():
    """测试avl插入"""
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    avl=AVLTree(inputs)
    res=avl.traverse()
    std=sorted(inputs)
    assert res==std
    for node in avl.traverse_nodes():
        assert abs(_BSTreeNode.get_h(node.lch)-_BSTreeNode.get_h(node.rch))<=1
        assert node.get_root()==avl.root
def test_avl_remove():
    """测试avl删除"""
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    avl=AVLTree(inputs)
    random.shuffle(inputs)
    for i in range(input_num):
        avl.remove(inputs[i])
        res=avl.traverse()
        std=sorted(inputs[i+1:])
        assert res==std
        for node in avl.traverse_nodes():
            assert abs(_BSTreeNode.get_h(node.lch)-_BSTreeNode.get_h(node.rch))<=1
            assert node.get_root()==avl.root

def random_test_segtree():
    """测试线段树合并区间（随机）"""
    intvs=[]
    limits=(0,10000,1000)
    random.seed(0)
    for _ in range(input_num):
        l=random.random()*(limits[1]-limits[0])+limits[0]
        # r=random.random()*(limits[1]-limits[0])+limits[0]
        r=l+1000
        h=random.random()*limits[2]
        intvs.append(Interval1d(l,r,h))
    # input_num=100000
    # intvs=[Interval1d(l,l+input_num+1,l) for l in range(input_num)]
    segtree=SegmentTree(intvs)
    merged_intvs=segtree.get_united_leaves()

    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    for _,intv in enumerate(intvs):
        plt.plot([intv.l,intv.r],[intv.value,intv.value])
    plt.subplot(2,1,2)
    for _,intv in enumerate(merged_intvs):
        plt.plot([intv.l,intv.r],[intv.value,intv.value])
    plt.show()

    print(f"{len(intvs)} lines before")
    print(f"{len(merged_intvs)} lines after")
    # write_stdout(intvs,SEG_TREE,"case_3")
    # write_stdout(merged_intvs,SEG_TREE,"out_3")

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,SEG_TREE[1]+1)],
    ids=[f"case_{i}" for i in range(1,SEG_TREE[1]+1)],
)
def test_segtree(case):
    """测试线段树合并区间"""
    intvs=read_case(SEG_TREE,case["in"])
    segtree=SegmentTree(intvs)
    merged_intvs=segtree.get_united_leaves()
    if __name__=="__main__":
        import matplotlib.pyplot as plt
        plt.subplot(2,1,1)
        for _,intv in enumerate(intvs):
            plt.plot([intv.l,intv.r],[intv.value,intv.value])
        plt.subplot(2,1,2)
        for _,intv in enumerate(merged_intvs):
            plt.plot([intv.l,intv.r],[intv.value,intv.value])
        plt.show()  
        # write_stdout(merged_intvs,SEG_TREE,f"out_{i}") 
    else:
        std_out=read_case(SEG_TREE,case["out"])
        assert merged_intvs==std_out

def random_test_kdtree():
    """随机测试kd树"""
    import matplotlib.pyplot as plt
    import math,numpy as np
    from lib.utils import Timer
    # Timer.enable()
    random.seed(0)

    n=10000
    limits=1000
    nodes=[Node(random.random()*limits,random.random()*limits) for i in range(n)]
    with Timer(tag="kdtree"):
        kdtree=KDTree(nodes)

    qx,qy=[random.random()*limits,random.random()*limits],[random.random()*limits,random.random()*limits]
    qbox=Box(min(qx),min(qy),max(qx),max(qy))
    qcenter=Node(random.random()*limits,random.random()*limits)
    qradius=random.random()*limits/5
    qcir=Circle(qcenter,qradius)
    
    plt.scatter([node.x for node in nodes],[node.y for node in nodes],color='k',alpha=0.3)
    nodes=[]
    with Timer(tag="traverse"):
        kdtree._root.traverse(callback=lambda treenode:nodes.append(treenode))
    for node in nodes:
        if node.dim==0:
            x=[node.obj.x,node.obj.x]
            y=[max(node.space.miny,0),min(node.space.maxy,1000)]
        elif node.dim==1: 
            y=[node.obj.y,node.obj.y]
            x=[max(node.space.minx,0),min(node.space.maxx,1000)]
        plt.plot(x,y,color='k',alpha=0.3)

    plt.plot([qbox.minx,qbox.maxx,qbox.maxx,qbox.minx,qbox.minx],[qbox.miny,qbox.miny,qbox.maxy,qbox.maxy,qbox.miny])
    with Timer(tag="box_q"):
        hit_nodes=kdtree.query_box(qbox)
    plt.scatter([node.x for node in hit_nodes],[node.y for node in hit_nodes],color='r')
    
    
    t=np.arange(0,2*math.pi,0.01)
    x=np.cos(t)*qradius+qcenter.x
    y=np.sin(t)*qradius+qcenter.y
    plt.plot(x,y,color='b')
    with Timer(tag="circle_q"):
        hit_nodes=kdtree.query_circle(qcir)
    plt.scatter([node.x for node in hit_nodes],[node.y for node in hit_nodes],color='g')
    
    plt.show()

if __name__=="__main__":
    if 0: test_bst_insert()
    if 0: random_test_segtree()
    if 0: test_segtree(({"in":f"case_{1}","out":f"out_{1}"}))
    if 1: random_test_kdtree()

