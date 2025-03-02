import pytest
from lib.index import BSTree
import random

def test_bst():
    inputs=[random.randint(0,10000) for _ in range(10000)]
    bst=BSTree(inputs)
    res=bst.traverse()
    std=sorted(inputs)
    assert res==std
    for i in range(10000):
        node=bst.find(inputs[i])
        assert node is not None
    assert bst.find(-1) is None
    random.shuffle(inputs)
    for i in range(10000):
        bst.remove(inputs[i])
        res=bst.traverse()
        std=sorted(inputs[i+1:])
        assert res==std

test_bst()