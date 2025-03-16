import pytest
from lib.index import BST,_BSTNode,AVLTree,SegmentTree
from tests.utils import set_root_dir,read_case,write_stdout
from lib.domain import Domain1d
import random

set_root_dir("./tests/tree/")

SEG_TREE=("seg_tree/",5)

input_num=1000
def test_bst_insert():
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    bst=BST(inputs)
    res=bst.traverse()
    std=sorted(inputs)
    assert res==std
    for node in bst.traverse_nodes():
        assert node.get_root()==bst.root    
def test_bst_find():
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    bst=BST(inputs)
    for i in range(input_num):
        node=bst.find(inputs[i])
        assert node is not None
    assert bst.find(-1) is None
def test_bst_remove():
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    bst=BST(inputs)
    random.shuffle(inputs)
    for i in range(input_num):
        bst.remove(inputs[i])
        res=bst.traverse()
        std=sorted(inputs[i+1:])
        assert res==std
def test_l_rotete():
    inputs=[2,1,4,3,5]
    avl=AVLTree(inputs)
    avl.root=avl.root._l_rotate()
    assert avl.root.obj==4 and avl.traverse()==[1,2,3,4,5]
def test_r_rotete():
    inputs=[4,5,2,3,1]
    avl=AVLTree(inputs)
    avl.root=avl.root._r_rotate()
    assert avl.root.obj==2 and avl.traverse()==[1,2,3,4,5]    
def test_avl_insert():
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    avl=AVLTree(inputs)
    res=avl.traverse()
    std=sorted(inputs)
    assert res==std
    for node in avl.traverse_nodes():
        assert abs(_BSTNode.get_h(node.lch)-_BSTNode.get_h(node.rch))<=1
        assert node.get_root()==avl.root
def test_avl_remove():
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    avl=AVLTree(inputs)
    random.shuffle(inputs)
    for i in range(input_num):
        avl.remove(inputs[i])
        res=avl.traverse()
        std=sorted(inputs[i+1:])
        assert res==std
        for node in avl.traverse_nodes():
            assert abs(_BSTNode.get_h(node.lch)-_BSTNode.get_h(node.rch))<=1
            assert node.get_root()==avl.root

def random_test_segtree():
    doms=[]
    limits=(0,10000,1000)
    random.seed(0)
    for _ in range(input_num):
        l=random.random()*(limits[1]-limits[0])+limits[0]
        # r=random.random()*(limits[1]-limits[0])+limits[0]
        r=l+1000
        h=random.random()*limits[2]
        doms.append(Domain1d(l,r,h))
    # input_num=100000
    # doms=[Domain1d(l,l+input_num+1,l) for l in range(input_num)]
    segtree=SegmentTree(doms)
    merged_doms=segtree.get_united_leaves()

    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    for _,dom in enumerate(doms):
        plt.plot([dom.l,dom.r],[dom.value,dom.value])    
    plt.subplot(2,1,2)
    for _,dom in enumerate(merged_doms):
        plt.plot([dom.l,dom.r],[dom.value,dom.value])
    plt.show()        

    print(f"{len(doms)} lines before")
    print(f"{len(merged_doms)} lines after")
    # write_stdout(doms,SEG_TREE,"case_3")
    # write_stdout(merged_doms,SEG_TREE,"out_3")

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,SEG_TREE[1]+1)],
    ids=[f"case_{i}" for i in range(1,SEG_TREE[1]+1)],
)
def test_segtree(case):
    doms=read_case(SEG_TREE,case["in"])
    segtree=SegmentTree(doms)
    merged_doms=segtree.get_united_leaves()
    if __name__=="__main__":
        import matplotlib.pyplot as plt
        plt.subplot(2,1,1)
        for _,dom in enumerate(doms):
            plt.plot([dom.l,dom.r],[dom.value,dom.value])    
        plt.subplot(2,1,2)
        for _,dom in enumerate(merged_doms):
            plt.plot([dom.l,dom.r],[dom.value,dom.value])
        plt.show()  
        # write_stdout(merged_doms,SEG_TREE,f"out_{i}") 
    else:
        std_out=read_case(SEG_TREE,case["out"])
        assert merged_doms==std_out

if __name__=="__main__":
    if 0: random_test_segtree()
    if 0: test_segtree(({"in":f"case_{1}","out":f"out_{1}"}))