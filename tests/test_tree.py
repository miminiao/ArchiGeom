import pytest
from lib.index import BST,_BSTNode,AVLTree
import random

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
    input_num=1000
    inputs=[random.randint(0,input_num) for _ in range(input_num)]
    avl=AVLTree(inputs)
    res=avl.traverse()
    std=sorted(inputs)
    assert res==std
    for node in avl.traverse_nodes():
        assert abs(_BSTNode.get_h(node.lch)-_BSTNode.get_h(node.rch))<=1
        assert node.get_root()==avl.root
def test_avl_remove():
    input_num=1000
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
def test_segtree_insert():
    ...