"""测试自相交Loop处理; positive=False,ensure_valid=False"""

import pytest
import json
from lib.utils import Constant
from lib.geom import Node,LineSeg,Arc,Loop
from lib.geom_algo import SplitIntersectedLoopsAlgo, _find_or_insert_node

const=Constant.default()
# const=Constant("split_loop",tol_area=1e3,tol_dist=1e-2)


@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"case_{i}_out"} for i in range(1,26)],
    ids=[f"case_{i}" for i in range(1,26)],
)
def test_split_loops_algo(case):
    """测试自相交Loop处理"""
    with open(f"./test/split_loop/{case['in']}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    loops=[]
    for obj in j_obj:
        nodes,edges=[],[]
        if obj["object_name"]=="polyline":
            seg_num=len(obj["segments"]) if obj["is_closed"] else len(obj["segments"])-1
            for i in range(seg_num):
                seg=obj["segments"][i]
                next_seg=obj["segments"][(i+1)%len(obj["segments"])]
                x1,y1,_=seg["start_point"]
                x2,y2,_=next_seg["start_point"]
                lw=rw=seg["start_width"]/2
                bulge=seg["bulge"]
                s=Node(x1,y1)
                e=Node(x2,y2)
                if s.equals(e):continue
                s=_find_or_insert_node(s,nodes)
                e=_find_or_insert_node(e,nodes)
                if abs(bulge)<const.TOL_VAL:
                    edges.append(LineSeg(s,e))
                else:
                    edges.append(Arc(s,e,bulge))
        loops.append(Loop(edges))
    split_loops=SplitIntersectedLoopsAlgo(loops,False,False,const=const).get_result()
    split_loops=list(filter(lambda loop:abs(loop.area*2/loop.length)>1,split_loops))
    split_loops.sort(key=lambda loop:loop.area)

    with open(f"./test/split_loop/{case['out']}.json",'r',encoding="utf8") as f:
        std_out=json.load(f)
    assert len(split_loops)==len(std_out)
    for i in range(len(std_out)):
        assert abs(std_out[i]-split_loops[i].area)/abs(std_out[i])<0.001
