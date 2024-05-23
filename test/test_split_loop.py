"""测试自相交Loop处理; positive=False,ensure_valid=False"""

import pytest
import json
from lib.utils import Constant
from lib.geom_algo import SplitIntersectedLoopsAlgo
from tool.dwg_converter.json_parser import polyline_to_loop
const=Constant.default()
# const=Constant("split_loop",tol_area=1e3,tol_dist=1e-2)


@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"case_{i}_out"} for i in range(1,30)],
    ids=[f"case_{i}" for i in range(1,30)],
)
def test_split_loops_algo(case):
    """测试自相交Loop处理"""
    with open(f"./test/split_loop/{case['in']}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    loops=polyline_to_loop(j_obj)

    split_loops=SplitIntersectedLoopsAlgo(loops,False,False,const=const).get_result()
    split_loops=list(filter(lambda loop:abs(loop.area*2/loop.length)>1,split_loops))

    with open(f"./test/split_loop/{case['out']}.json",'r',encoding="utf8") as f:
        std_out=json.load(f)
    std_out=list(filter(lambda area:abs(area)>const.TOL_AREA,std_out))
    
    split_loops=list(filter(lambda loop:abs(loop.area)>const.TOL_AREA,split_loops))
    split_loops.sort(key=lambda loop:loop.area)
    
    assert len(split_loops)==len(std_out)
    for i in range(len(std_out)):
        assert (abs(std_out[i]-split_loops[i].area)/abs(std_out[i])<0.001
                or abs(std_out[i]-split_loops[i].area)<const.TOL_AREA)
