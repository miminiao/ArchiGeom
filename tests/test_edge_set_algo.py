"""测试Edge集合相关的操作"""

import pytest
from lib.utils import Constant as Const
from lib.geom_algo import BreakEdgeAlgo, MergeEdgeAlgo,FindOutlineAlgo
from lib.geom_plotter import CADPlotter
from tests.utils import read_case,write_stdout

ROOT="./tests/edge_set_algo/"

BREAK_EDGE=(ROOT+"break_edge/",6)
FIND_OUTLINE=(ROOT+"find_outline/",4)
MEARGE_EDGE=(ROOT+"merge_edge/",9)

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,BREAK_EDGE[1]+1)],
    ids=[f"case_{i}" for i in range(1,BREAK_EDGE[1]+1)],
)
def test_break_edge(case):
    """测试边打断"""
    inputs=read_case(BREAK_EDGE,case["in"],hook_mode="cad")
    res=BreakEdgeAlgo(inputs).get_result()
    if __name__=="__main__":
        CADPlotter.draw_geoms(res)
        # write_stdout(len(res),BREAK_EDGE,f"out_{i}") 
    else:
        std_out=read_case(BREAK_EDGE,case["out"])
        assert len(res)==std_out

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,MEARGE_EDGE[1]+1)],
    ids=[f"case_{i}" for i in range(1,MEARGE_EDGE[1]+1)],
)
def test_merge_edge(case):
    """测试边合并"""
    edges=read_case(MEARGE_EDGE,case["in"],hook_mode="cad")
    res=MergeEdgeAlgo(edges,break_at_intersections=False).get_result()
    if __name__=="__main__":
        CADPlotter.draw_geoms(res)
        # write_stdout(len(res),MEARGE_EDGE,f"out_{i}") 
    else:
        std_out=read_case(MEARGE_EDGE,case["out"])
        assert len(res)==std_out

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,FIND_OUTLINE[1]+1)],
    ids=[f"case_{i}" for i in range(1,FIND_OUTLINE[1]+1)],
)
def test_find_outline(case):
    """测试单连通分量外轮廓"""
    edges=read_case(FIND_OUTLINE,case["in"],hook_mode="cad")
    res=FindOutlineAlgo(edges).get_result()
    comp=Const.cmp_area
    if __name__=="__main__":
        CADPlotter.draw_geoms([res])
        # write_stdout([len(res),res.area],FIND_OUTLINE,f"out_{i}") 
    else:
        std_out=read_case(FIND_OUTLINE,case["out"])
        assert len(res)==std_out[0]
        assert comp(res.area,std_out[1])==0

if __name__=="__main__":
    if 1:
        for i in [3]:
        # for i in range(1,FIND_OUTLINE[1]+1):
            test_find_outline({"in":f"case_{i}","out":f"out_{i}"})
    if 0:
        for i in [5]:
        # for i in range(1,BREAK_EDGE[1]+1):
            test_break_edge({"in":f"case_{i}","out":f"out_{i}"})            
    if 0:
        for i in [5]:
        # for i in range(1,MEARGE_EDGE[1]+1):
            test_merge_edge({"in":f"case_{i}","out":f"out_{i}"})       