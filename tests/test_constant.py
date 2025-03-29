"""常量类测试"""

import json
from lib.utils import Constant as Const, ComparerInjector as CmpInj

with open("env.json") as f:
    env=json.load(f)
def test_constant_context():
    """常量上下文测试"""
    std=env["CONSTANTS"]
    assert float(std["MAX_VAL"])==Const.MAX_VAL
    assert float(std["TOL_VAL"])==Const.TOL_VAL
    assert float(std["TOL_DIST"])==Const.TOL_DIST
    assert float(std["TOL_AREA"])==Const.TOL_AREA
    assert float(std["TOL_ANG"])==Const.TOL_ANG    
    with Const(MAX_VAL=1,TOL_VAL=2,TOL_DIST=3,TOL_AREA=4,TOL_ANG=5):
        assert Const.MAX_VAL==1
        assert Const.TOL_VAL==2
        assert Const.TOL_DIST==3
        assert Const.TOL_AREA==4
        assert Const.TOL_ANG==5
        with Const(MAX_VAL=10,TOL_VAL=20,TOL_DIST=30,TOL_AREA=40,TOL_ANG=50):
            assert Const.MAX_VAL==10
            assert Const.TOL_VAL==20
            assert Const.TOL_DIST==30
            assert Const.TOL_AREA==40
            assert Const.TOL_ANG==50
        assert Const.MAX_VAL==1
        assert Const.TOL_VAL==2
        assert Const.TOL_DIST==3
        assert Const.TOL_AREA==4
        assert Const.TOL_ANG==5
    assert float(std["MAX_VAL"])==Const.MAX_VAL
    assert float(std["TOL_VAL"])==Const.TOL_VAL
    assert float(std["TOL_DIST"])==Const.TOL_DIST
    assert float(std["TOL_AREA"])==Const.TOL_AREA
    assert float(std["TOL_ANG"])==Const.TOL_ANG            
def test_comparer_injector():
    """比较函数注入上下文测试"""
    class Test:
        def __init__(self,value): 
            self.value=value
    a=Test(1)
    b=Test(2)
    try:
        if a<b: assert False
    except: assert True
    cmp=lambda x,y: Const.cmp_val(x.value,y.value)
    with CmpInj(Test,cmp,override_ops=True):
        assert Test._cmp(a,b)<0
        assert a<b
    try:
        if a<b: assert False
    except: assert True

    