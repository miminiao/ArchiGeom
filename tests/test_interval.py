"""测试区间操作"""

import pytest
from lib.interval import Interval1d, MultiInterval1d
from tests.utils import read_case,write_stdout
import random

ROOT="./tests/interval/"

INTERVAL_UNION=(ROOT+"interval_union/",5)

@pytest.mark.parametrize('case',[
    {"in":[[0,1,1],[0.5,1.5,2]],"out":[[0,0.5,1],[0.5,1.5,2]]},
    {"in":[[0.5,1.5,2],[0,1,1]],"out":[[0,0.5,1],[0.5,1.5,2]]},
    {"in":[[0.5,1.5,2],[0,1,2]],"out":[[0,1.5,2]]},
    {"in":[[0,1,1],[1.1,1.5,2]],"out":[[0,1,1],[1.1,1.5,2]]},
    {"in":[[0,2,1],[0.5,1.5,2]],"out":[[0,0.5,1],[0.5,1.5,2],[1.5,2,1]]},
    {"in":[[0,2,2],[0.5,1.5,1]],"out":[[0,2,2]]},
    {"in":[[0.5,1.5,2],[0,2,1]],"out":[[0,0.5,1],[0.5,1.5,2],[1.5,2,1]]},
    {"in":[[0.5,1.5,1],[0,2,2]],"out":[[0,2,2]]},
    {"in":[[0,1,2],[1,2,2]],"out":[[0,2,2]]},
    ])
def test_add_higher(case):
    """测试单区间并集(取高)"""
    a=Interval1d(*case["in"][0])
    b=Interval1d(*case["in"][1])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
    else:
        c=MultiInterval1d([Interval1d(*intv) for intv in case["out"]])
    assert a+b==c

@pytest.mark.parametrize('case',[
    {"in":[[0,1,1],[0.5,1.5,2]],"out":[[0,1,1],[1,1.5,2]]},
    {"in":[[0.5,1.5,2],[0,1,1]],"out":[[0,1,1],[1,1.5,2]]},
    {"in":[[0.5,1.5,2],[0,1,2]],"out":[[0,1.5,2]]},
    {"in":[[0,1,1],[1.1,1.5,2]],"out":[[0,1,1],[1.1,1.5,2]]},
    {"in":[[0,2,1],[0.5,1.5,2]],"out":[[0,2,1]]},
    {"in":[[0,2,2],[0.5,1.5,1]],"out":[[0,0.5,2],[0.5,1.5,1],[1.5,2,2]]},
    {"in":[[0.5,1.5,2],[0,2,1]],"out":[[0,2,1]]},
    {"in":[[0.5,1.5,1],[0,2,2]],"out":[[0,0.5,2],[0.5,1.5,1],[1.5,2,2]]},
    {"in":[[0,1,2],[1,2,2]],"out":[[0,2,2]]},
    ])
def test_add_lower(case):
    """测试单区间并集(取矮)"""
    a=Interval1d(*case["in"][0])
    a.value=-a.value
    b=Interval1d(*case["in"][1])
    b.value=-b.value
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
        c.value=-c.value
    else:
        c=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["out"]])
    assert a+b==c

@pytest.mark.parametrize('case',[
    {"in":[[0,1,1],[0.5,1.5,2]],"out":[[0,0.5,1]]},
    {"in":[[0.5,1.5,2],[0,1,1]],"out":[[1,1.5,2]]},
    {"in":[[0.5,1.5,2],[0,1,2]],"out":[[1,1.5,2]]},
    {"in":[[0,1,1],[1.1,1.5,2]],"out":[[0,1,1]]},
    {"in":[[0,2,1],[0.5,1.5,2]],"out":[[0,0.5,1],[1.5,2,1]]},
    {"in":[[0,2,2],[0.5,1.5,1]],"out":[[0,0.5,2],[1.5,2,2]]},
    {"in":[[0.5,1.5,2],[0,2,1]],"out":[[1,0,0]]},
    {"in":[[0.5,1.5,1],[0,2,2]],"out":[[1,0,0]]},
    ])
def test_sub(case):
    """测试单区间差集"""
    a=Interval1d(*case["in"][0])
    b=Interval1d(*case["in"][1])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
    else:
        c=MultiInterval1d([Interval1d(*intv) for intv in case["out"]])
    assert a-b==c

@pytest.mark.parametrize('case',[
    {"in":[[0,1,1],[0.5,1.5,2]],"out":[[0.5,1,1]]},
    {"in":[[0.5,1.5,2],[0,1,1]],"out":[[0.5,1,1]]},
    {"in":[[0.5,1.5,2],[0,1,2]],"out":[[0.5,1,2]]},
    {"in":[[0,1,1],[1.1,1.5,2]],"out":[[1,0,0]]},
    {"in":[[0,2,1],[0.5,1.5,2]],"out":[[0.5,1.5,1]]},
    {"in":[[0,2,2],[0.5,1.5,1]],"out":[[0.5,1.5,1]]},
    {"in":[[0.5,1.5,2],[0,2,1]],"out":[[0.5,1.5,1]]},
    {"in":[[0.5,1.5,1],[0,2,2]],"out":[[0.5,1.5,1]]},
    {"in":[[0,1,2],[1,2,2]],"out":[[1,0,0]]},    
    ])
def test_mul_higher(case):
    """测试单区间交集(取高)"""
    a=Interval1d(*case["in"][0])
    b=Interval1d(*case["in"][1])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
    else:
        c=MultiInterval1d([Interval1d(*intv) for intv in case["out"]])
    assert a*b==c

@pytest.mark.parametrize('case',[
    {"in":[[0,1,1],[0.5,1.5,2]],"out":[[0.5,1,2]]},
    {"in":[[0.5,1.5,2],[0,1,1]],"out":[[0.5,1,2]]},
    {"in":[[0.5,1.5,2],[0,1,2]],"out":[[0.5,1,2]]},
    {"in":[[0,1,1],[1.1,1.5,2]],"out":[[1,0,0]]},
    {"in":[[0,2,1],[0.5,1.5,2]],"out":[[0.5,1.5,2]]},
    {"in":[[0,2,2],[0.5,1.5,1]],"out":[[0.5,1.5,2]]},
    {"in":[[0.5,1.5,2],[0,2,1]],"out":[[0.5,1.5,2]]},
    {"in":[[0.5,1.5,1],[0,2,2]],"out":[[0.5,1.5,2]]},
    ])
def test_mul_lower(case):
    """测试单区间交集(取矮)"""
    a=Interval1d(*case["in"][0])
    a.value=-a.value
    b=Interval1d(*case["in"][1])
    b.value=-b.value
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
        c.value=-c.value
    else:
        c=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["out"]])
    assert a*b==c

@pytest.mark.parametrize('case',[
    {"in":[[[0,1,1],[1.5,3,1],[3.5,4.5,1]],[[0.5,2,2],[2.5,4,2]]],"out":[[0,0.5,1],[0.5,2,2],[2,2.5,1],[2.5,4,2],[4,4.5,1]]},
    {"in":[[[1.5,2,2],[4,4.5,2]],[[0,1,1],[2.5,3.5,1]]],"out":[[0,1,1],[1.5,2,2],[2.5,3.5,1],[4,4.5,2]]},
    {"in":[[[0,1.5,2],[3,4.5,2]],[[0,0.5,1],[1,2,1],[2.5,3,1],[3.5,4,1]]],"out":[[0,1.5,2],[1.5,2,1],[2.5,3,1],[3,4.5,2]]},
    {"in":[[[0,3.5,1],[4,4.5,2]],[[0.5,1,2],[1.5,2,2],[2.5,3,1]]],"out":[[0,0.5,1],[0.5,1,2],[1,1.5,1],[1.5,2,2],[2,3.5,1],[4,4.5,2]]},
    {"in":[[[0,2,2],[2.5,3,2],[3.5,4,2]],[[0.5,1.5,2],[2,2.5,2],[3,3.5,2],[4,4.5,2]]],"out":[[0,4.5,2]]},
    {"in":[[[0,2,2],[2.5,3.5,1]],[[0.5,1.5,1],[2,4,2]]],"out":[[0,4,2]]},
    ])
def test_multi_add_higher(case):
    """测试多区间并集(取高)"""
    a=MultiInterval1d([Interval1d(*intv) for intv in case["in"][0]])
    b=MultiInterval1d([Interval1d(*intv) for intv in case["in"][1]])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
    else:
        c=MultiInterval1d([Interval1d(*intv) for intv in case["out"]])
    assert a+b==c

@pytest.mark.parametrize('case',[
    {"in":[[[0,1,1],[1.5,3,1],[3.5,4.5,1]],[[0.5,2,2],[2.5,4,2]]],"out":[[0,1,1],[1,1.5,2],[1.5,3,1],[3,3.5,2],[3.5,4.5,1]]},
    {"in":[[[1.5,2,2],[4,4.5,2]],[[0,1,1],[2.5,3.5,1]]],"out":[[0,1,1],[1.5,2,2],[2.5,3.5,1],[4,4.5,2]]},
    {"in":[[[0,1.5,2],[3,4.5,2]],[[0,0.5,1],[1,2,1],[2.5,3,1],[3.5,4,1]]],"out":[[0,0.5,1],[0.5,1,2],[1,2,1],[2.5,3,1],[3,3.5,2],[3.5,4,1],[4,4.5,2]]},
    {"in":[[[0,3.5,1],[4,4.5,2]],[[0.5,1,2],[1.5,2,2],[2.5,3,1]]],"out":[[0,3.5,1],[4,4.5,2]]},
    {"in":[[[0,2,2],[2.5,3,2],[3.5,4,2]],[[0.5,1.5,2],[2,2.5,2],[3,3.5,2],[4,4.5,2]]],"out":[[0,4.5,2]]},
    {"in":[[[0,2,2],[2.5,3.5,1]],[[0.5,1.5,1],[2,4,2]]],"out":[[0,0.5,2],[0.5,1.5,1],[1.5,2.5,2],[2.5,3.5,1],[3.5,4,2]]},
    ])
def test_multi_add_lower(case):
    """测试多区间并集(取矮)"""
    a=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["in"][0]])
    b=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["in"][1]])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
        c.value=-c.value
    else:
        c=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["out"]])
    assert a+b==c

@pytest.mark.parametrize('case',[
    {"in":[[[0,1,1],[1.5,3,1],[3.5,4.5,1]],[[0.5,2,2],[2.5,4,2]]],"out":[[0,0.5,1],[2,2.5,1],[4,4.5,1]]},
    {"in":[[[1.5,2,2],[4,4.5,2]],[[0,1,1],[2.5,3.5,1]]],"out":[[1.5,2,2],[4,4.5,2]]},
    {"in":[[[0,1.5,2],[3,4.5,2]],[[0,0.5,1],[1,2,1],[2.5,3,1],[3.5,4,1]]],"out":[[0.5,1,2],[3,3.5,2],[4,4.5,2]]},
    {"in":[[[0,3.5,1],[4,4.5,2]],[[0.5,1,2],[1.5,2,2],[2.5,3,1]]],"out":[[0,0.5,1],[1,1.5,1],[2,2.5,1],[3,3.5,1],[4,4.5,2]]},
    {"in":[[[0,2,2],[2.5,3,2],[3.5,4,2]],[[0.5,1.5,2],[2,2.5,2],[3,3.5,2],[4,4.5,2]]],"out":[[0,0.5,2],[1.5,2,2],[2.5,3,2],[3.5,4,2]]},
    {"in":[[[0,2,2],[2.5,3.5,1]],[[0.5,1.5,1],[2,4,2]]],"out":[[0,0.5,2],[1.5,2,2]]},
    ])
def test_multi_sub(case):
    """测试多区间差集"""
    a=MultiInterval1d([Interval1d(*intv) for intv in case["in"][0]])
    b=MultiInterval1d([Interval1d(*intv) for intv in case["in"][1]])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
    else:
        c=MultiInterval1d([Interval1d(*intv) for intv in case["out"]])
    assert a-b==c

@pytest.mark.parametrize('case',[
    {"in":[[[0,1,1],[1.5,3,1],[3.5,4.5,1]],[[0.5,2,2],[2.5,4,2]]],"out":[[0.5,1,1],[1.5,2,1],[2.5,3,1],[3.5,4,1]]},
    {"in":[[[1.5,2,2],[4,4.5,2]],[[0,1,1],[2.5,3.5,1]]],"out":[[1,0,0]]},
    {"in":[[[0,1.5,2],[3,4.5,2]],[[0,0.5,1],[1,2,1],[2.5,3,1],[3.5,4,1]]],"out":[[0,0.5,1],[1,1.5,1],[3.5,4,1]]},
    {"in":[[[0,3.5,1],[4,4.5,2]],[[0.5,1,2],[1.5,2,2],[2.5,3,1]]],"out":[[0.5,1,1],[1.5,2,1],[2.5,3,1]]},
    {"in":[[[0,2,2],[2.5,3,2],[3.5,4,2]],[[0.5,1.5,2],[2,2.5,2],[3,3.5,2],[4,4.5,2]]],"out":[[0.5,1.5,2]]},
    {"in":[[[0,2,2],[2.5,3.5,1]],[[0.5,1.5,1],[2,4,2]]],"out":[[0.5,1.5,1],[2.5,3.5,1]]},
    ])
def test_multi_mul_higher(case):
    """测试多区间交集(取高)"""
    a=MultiInterval1d([Interval1d(*intv) for intv in case["in"][0]])
    b=MultiInterval1d([Interval1d(*intv) for intv in case["in"][1]])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
    else:
        c=MultiInterval1d([Interval1d(*intv) for intv in case["out"]])
    assert a*b==c

@pytest.mark.parametrize('case',[
    {"in":[[[0,1,1],[1.5,3,1],[3.5,4.5,1]],[[0.5,2,2],[2.5,4,2]]],"out":[[0.5,1,2],[1.5,2,2],[2.5,3,2],[3.5,4,2]]},
    {"in":[[[1.5,2,2],[4,4.5,2]],[[0,1,1],[2.5,3.5,1]]],"out":[[1,0,0]]},
    {"in":[[[0,1.5,2],[3,4.5,2]],[[0,0.5,1],[1,2,1],[2.5,3,1],[3.5,4,1]]],"out":[[0,0.5,2],[1,1.5,2],[3.5,4,2]]},
    {"in":[[[0,3.5,1],[4,4.5,2]],[[0.5,1,2],[1.5,2,2],[2.5,3,1]]],"out":[[0.5,1,2],[1.5,2,2],[2.5,3,1]]},
    {"in":[[[0,2,2],[2.5,3,2],[3.5,4,2]],[[0.5,1.5,2],[2,2.5,2],[3,3.5,2],[4,4.5,2]]],"out":[[0.5,1.5,2]]},
    {"in":[[[0,2,2],[2.5,3.5,1]],[[0.5,1.5,1],[2,4,2]]],"out":[[0.5,1.5,2],[2.5,3.5,2]]},    
    ])
def test_multi_mul_lower(case):
    """测试多区间交集(取矮)"""
    a=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["in"][0]])
    b=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["in"][1]])
    if len(case["out"])==1:
        c=Interval1d(*case["out"][0])
        c.value=-c.value
    else:
        c=MultiInterval1d([Interval1d(l,r,-v) for l,r,v in case["out"]])
    assert a*b==c

@pytest.mark.parametrize('case',[
    {"in":[[[0,1,1],[2,4,1],[5,7,1]],[[3,6,2]]],"out":True},
    {"in":[[[3,6,2]],[[0,1,1],[2,4,1],[5,7,1]]],"out":True},
    {"in":[[[0,1,1],[4,5,1]],[[2,3,2],[6,7,2]]],"out":False},
    {"in":[[[2,3,2],[6,7,2]],[[0,1,1],[4,5,1]]],"out":False},
    {"in":[[[0,1,1],[2,3,1]],[[4,5,2],[6,7,2]]],"out":False},
    {"in":[[[4,5,2],[6,7,2]],[[0,1,1],[2,3,1]]],"out":False},
    ])
# @pytest.mark.timeout(5)
def test_multi_overlap(case):
    """测试多区间重叠判断"""
    a=MultiInterval1d([Interval1d(*intv) for intv in case["in"][0]])
    b=MultiInterval1d([Interval1d(*intv) for intv in case["in"][1]])
    assert a.is_overlap(b,include_endpoints=True)==case["out"]

def random_test_intv_union():
    """测试n个区间合并（随机）"""
    intvs=[]
    limits=(0,10000,1000)
    random.seed(1)
    for _ in range(1000):
        l=random.random()*(limits[1]-limits[0])+limits[0]
        r=l+1000
        h=random.random()*limits[2]
        intvs.append(Interval1d(l,r,h))
    # merged_intvs=Interval1d.union(intvs)
    merged_intvs=Interval1d.union(intvs,ignore_value=False)

    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    for _,intv in enumerate(intvs):
        plt.plot([intv.l,intv.r],[intv.value,intv.value]) 
        plt.plot([intv.l,intv.l],[intv.value,0],'b--',linewidth=1)
        plt.plot([intv.r,intv.r],[intv.value,0],'b--',linewidth=1)
    plt.subplot(2,1,2)
    if isinstance(merged_intvs,Interval1d): merged_intvs=MultiInterval1d([merged_intvs])
    for _,intv in enumerate(merged_intvs._items):
        plt.plot([intv.l,intv.r],[intv.value,intv.value])
    plt.show()        

    print(f"{len(intvs)} lines before")
    print(f"{len(merged_intvs)} lines after")

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,INTERVAL_UNION[1]+1)],
    ids=[f"case_{i}" for i in range(1,INTERVAL_UNION[1]+1)],
)
def test_intv_union(case):
    """测试n个区间合并"""
    intvs=read_case(INTERVAL_UNION,case["in"])
    merged_intvs=Interval1d.union(intvs,ignore_value=False)
    if __name__=="__main__":
        import matplotlib.pyplot as plt
        plt.subplot(2,1,1)
        for _,intv in enumerate(intvs):
            plt.plot([intv.l,intv.r],[intv.value,intv.value])
        plt.subplot(2,1,2)
        for _,intv in enumerate(merged_intvs):
            plt.plot([intv.l,intv.r],[intv.value,intv.value])
        plt.show()  
        # write_stdout(merged_intvs,INTERVAL_UNION,f"out_{i}") 
    else:
        std_out=read_case(INTERVAL_UNION,case["out"])
        assert merged_intvs._items==std_out

if __name__=="__main__":
    if 0: random_test_intv_union()
    if 0: test_intv_union(({"in":f"case_{1}","out":f"out_{1}"}))
