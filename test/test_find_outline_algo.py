"""测试最大外轮廓"""

import pytest
import json
from lib.utils import Constant; const=Constant.default()
from lib.geom import Node,LineSeg,Arc
from lib.geom_algo import FindOutlineAlgo

@pytest.mark.parametrize('case',[
    {"in":"case_1","out":[10,16140606.985840075]},
    {"in":"case_2","out":[139,1398016163.7705956]},
    {"in":"case_3","out":[458,6099487468.004033]},
    ])
def test_find_single_outline(case):
    """单连通分量最大外轮廓"""
    with open(f"./test/find_outline/{case['in']}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges=[]
    for ent in j_obj:
        x1,y1,_=ent["start_point"]
        x2,y2,_=ent["end_point"]
        s=Node(x1,y1)
        e=Node(x2,y2)
        if s.equals(e):continue
        if ent["object_name"]=="line":
            edges.append(LineSeg(s,e))
        elif ent["object_name"]=="arc":
            edges.append(Arc(s,e,ent["bulge"]))
    outline=FindOutlineAlgo(edges).get_result()
    assert len(outline.edges)==case["out"][0]
    assert abs(outline.get_area()-case["out"][1])<const.TOL_AREA