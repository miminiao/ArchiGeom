"""测试线段处理"""

import pytest
import json
from lib.utils import Constant; const=Constant.default()
from lib.geom import Node,Edge,LineSeg,Arc
from lib.geom_algo import BreakEdgeAlgo,MergeLineAlgo

@pytest.mark.parametrize('case',[
    {"in":"case_1","out":51768},
    ])
def test_break_line_algo(case):
    """打断"""
    with open(f"./test/line_set/{case['in']}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    lines=[]
    for obj in j_obj:
        if obj["object_name"]!="line":continue
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(LineSeg(s,e))
    broken_lines=BreakEdgeAlgo([lines]).get_result()[0]
    assert len(broken_lines)>=case["out"]

@pytest.mark.parametrize('case',[
    {"in":"case_1","out":19671},
    ])
def test_merge_line_algo_no_break(case):
    """合并，忽略交点"""
    with open(f"./test/line_set/{case['in']}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    lines=[]
    for obj in j_obj:
        if obj["object_name"]!="line":continue
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(Edge(s,e))
    merged_lines=MergeLineAlgo(lines,preserve_intersections=False).get_result()
    assert len(merged_lines)<=case["out"]

@pytest.mark.parametrize('case',[
    {"in":"case_1","out":49447},
    ])
def test_merge_line_algo_break(case):
    """合并，交点处打断"""
    with open(f"./test/line_set/{case['in']}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    lines=[]
    for obj in j_obj:
        if obj["object_name"]!="line":continue
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(Edge(s,e))
    merged_lines=MergeLineAlgo(lines,preserve_intersections=True).get_result()
    assert len(merged_lines)<=case["out"]
