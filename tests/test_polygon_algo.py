"""测试Polygon相关的操作"""

import pytest
from lib.geom import Node,Arc,Edge,Polyedge,Loop,Polygon
from lib.geom_algo import BooleanOperation
from lib.geom_plotter import CADPlotter
from tests.utils import read_case,write_stdout
from lib.utils import Constant as Const
from lib.index import STRTree

ROOT="./tests/loop_polygon_algo/"

POLYGON_UNION=(ROOT+"polygon_union/",16)
POLYGON_DIFF=(ROOT+"polygon_diff/",16)
POLYGON_INTER=(ROOT+"polygon_inter/",16)

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,POLYGON_UNION[1]+1)],
    ids=[f"case_{i}" for i in range(1,POLYGON_UNION[1]+1)],
)
def test_polygon_union(case):
    """测试Polygon布尔并"""
    inputs=read_case(POLYGON_UNION,case["in"],hook_mode="cad")
    polygons=[g for g in inputs if isinstance(g,Polygon)]
    res=BooleanOperation.union(polygons)
    if __name__=="__main__":
        CADPlotter.draw_geoms(res)
        # write_stdout(res,POLYGON_UNION,f"out_{i}")
    else:
        std_out=read_case(POLYGON_UNION,case["out"])
        assert len(res)==len(std_out)
        std_tree=STRTree(std_out)
        assert all([_find_identical_polygon(geom,std_tree) for geom in res])

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,POLYGON_DIFF[1]+1)],
    ids=[f"case_{i}" for i in range(1,POLYGON_DIFF[1]+1)],
)
def test_polygon_diff(case):
    """测试Polygon布尔差"""
    inputs=read_case(POLYGON_DIFF,case["in"],hook_mode="cad")
    res=BooleanOperation.difference(inputs[0],inputs[1])
    if __name__=="__main__":
        CADPlotter.draw_geoms(res)
        # write_stdout(res,POLYGON_DIFF,f"out_{i}")
    else:
        std_out=read_case(POLYGON_DIFF,case["out"])
        assert len(res)==len(std_out)
        std_tree=STRTree(std_out)
        assert all([_find_identical_polygon(geom,std_tree) for geom in res])

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,POLYGON_INTER[1]+1)],
    ids=[f"case_{i}" for i in range(1,POLYGON_INTER[1]+1)],
)
def test_polygon_inter(case):
    """测试Polygon布尔交"""
    inputs=read_case(POLYGON_INTER,case["in"],hook_mode="cad")
    polygons=[g for g in inputs if isinstance(g,Polygon)]
    res=BooleanOperation.intersection(polygons)    
    if __name__=="__main__":
        CADPlotter.draw_geoms(res)
        # write_stdout(res,POLYGON_INTER,f"out_{i}")
    else:
        std_out=read_case(POLYGON_INTER,case["out"])
        assert len(res)==len(std_out)
        std_tree=STRTree(std_out)
        assert all([_find_identical_polygon(geom,std_tree) for geom in res])

def _find_identical_polygon(polygon:Polygon,std_out:STRTree[Polygon])->bool:
    neighbors=std_out.query(polygon.get_mbb())
    cmp=Const.cmp_area
    for std_polygon in neighbors:
        if cmp(std_polygon.area,polygon.area)==0 and std_polygon.covers(polygon) and polygon.covers(std_polygon):
            return True
    return False

# -------------------------------------------------------------------------------------

if __name__=="__main__":
    if 0:
        for i in [16]:
        # for i in range(1,POLYGON_UNION[1]+1):
            test_polygon_union({"in":f"case_{i}","out":f"out_{i}"})     
    if 0:
        # for i in [1]:
        for i in range(1,POLYGON_DIFF[1]+1):
            test_polygon_diff({"in":f"case_{i}","out":f"out_{i}"})     
    if 0:
        # for i in [1]:
        for i in range(1,POLYGON_INTER[1]+1):
            test_polygon_inter({"in":f"case_{i}","out":f"out_{i}"})     