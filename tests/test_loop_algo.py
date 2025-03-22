"""测试Loop相关的操作"""

import pytest
from lib.geom import Node,Arc,Edge,Polyedge,Loop
from lib.geom_algo import FindLoopAlgo
from lib.geom_plotter import CADPlotter
from tests.utils import read_case,write_stdout
from lib.utils import Constant as Const
from lib.index import STRTree

ROOT="./tests/loop_polygon_algo/"

COVER_NODE=(ROOT+"loop_covers_node/",7)
COVER_EDGE=(ROOT+"loop_covers_edge/",8)
COVER_LOOP=(ROOT+"loop_covers_loop/",13)
REBUILD_LOOP=(ROOT+"rebuild_loop/",15)
REBUILD_AND_CANCEL_LOOP=(ROOT+"rebuild_and_cancel_loop/",29)
FIND_LOOP=(ROOT+"find_loop/",10)

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,COVER_NODE[1]+1)],
    ids=[f"case_{i}" for i in range(1,COVER_NODE[1]+1)],
)
def test_loop_covers_node(case):
    inputs=read_case(COVER_NODE,case["in"],hook_mode="cad")
    loop=[g for g in inputs if isinstance(g,(Polyedge,Loop))][0]
    points=[g.center for g in inputs if isinstance(g,Arc) and g.angles[0]==0]    
    covered_idx,other_idx=_loop_covers_node(loop,points)
    if __name__=="__main__":
        covered=[points[i] for i in covered_idx]
        others=[points[i] for i in other_idx]
        CADPlotter.draw_geoms([loop])
        CADPlotter.draw_geoms(covered,color=3)
        CADPlotter.draw_geoms(others,color=1)
        # write_stdout(covered_idx,COVER_NODE,f"out_{i}")
    else:
        std_out=read_case(COVER_NODE,case["out"])
        assert sorted(covered_idx)==sorted(std_out)

def _loop_covers_node(loop:Loop,points:list[Node])->tuple[list[int],list[int]]:
    covered_idx,other_idx=[],[]
    for i,p in enumerate(points):
        if loop._covers_node(p): 
            covered_idx.append(i)
        else: other_idx.append(i)
    return covered_idx,other_idx

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,COVER_EDGE[1]+1)],
    ids=[f"case_{i}" for i in range(1,COVER_EDGE[1]+1)],
)
def test_loop_covers_edge(case):
    inputs=read_case(COVER_EDGE,case["in"],hook_mode="cad")
    loop=[g for g in inputs if isinstance(g,(Polyedge,Loop))][0]
    edges=[g for g in inputs if isinstance(g,Edge)]
    covered_idx,other_idx=_loop_covers_edge(loop,edges)
    if __name__=="__main__":
        covered=[edges[i] for i in covered_idx]
        others=[edges[i] for i in other_idx]            
        CADPlotter.draw_geoms([loop])
        CADPlotter.draw_geoms(covered,color=3)
        CADPlotter.draw_geoms(others,color=1)        
        # write_stdout(covered_idx,COVER_EDGE,f"out_{i}")
    else:
        std_out=read_case(COVER_EDGE,case["out"])        
        assert sorted(covered_idx)==sorted(std_out)

def _loop_covers_edge(loop:Loop,edges:list[Edge])->tuple[list[int],list[int]]: 
    covered_idx,other_idx=[],[]
    for i,p in enumerate(edges): 
        if loop._covers_edge(p): 
            covered_idx.append(i)
        else: other_idx.append(i)
    return covered_idx,other_idx

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,COVER_LOOP[1]+1)],
    ids=[f"case_{i}" for i in range(1,COVER_LOOP[1]+1)],
)
def test_loop_covers_loop(case)->None:
    inputs=read_case(COVER_LOOP,case["in"],hook_mode="cad")
    loops=[g for g in inputs if isinstance(g,Loop)]
    coveres=_loop_covers_loop(loops)
    if __name__=="__main__":
        CADPlotter.draw_geoms([loops[0]],color=3 if coveres[0] else 1)        
        CADPlotter.draw_geoms([loops[1]],color=3 if coveres[1] else 1)    
        # write_stdout(covered,COVER_LOOP,f"out_{i}")
    else:
        std_out=read_case(COVER_LOOP,case["out"])
        assert coveres==std_out

def _loop_covers_loop(loops:list[Loop]):
    return [loops[0]._covers_polyedge(loops[1]),
            loops[1]._covers_polyedge(loops[0])]

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,REBUILD_LOOP[1]+1)],
    ids=[f"case_{i}" for i in range(1,REBUILD_LOOP[1]+1)],
)
def test_rebuild_loop(case)->None:
    """重建Loop有向图拓扑，保留反向边"""
    inputs=read_case(REBUILD_LOOP,case["in"],hook_mode="cad")
    loops=[g for g in inputs if isinstance(g,Loop)]
    res=_rebuild_loop(loops)
    if __name__=="__main__":
        CADPlotter.draw_geoms(res)
        # write_stdout(res,REBUILD_LOOP,f"out_{i}")
    else:
        std_out=read_case(REBUILD_LOOP,case["out"])      
        assert len(res)==len(std_out)
        std_tree=STRTree(std_out)  
        assert all([_find_identical_loop(geom,std_tree) for geom in res])

def _find_identical_loop(loop:Loop,std_out:STRTree[Loop])->bool:
    neighbors=std_out.query(loop.get_mbb())
    cmp=Const.cmp_area
    for std_loop in neighbors:
        if cmp(std_loop.area,loop.area)==0 and std_loop.covers(loop) and loop.covers(std_loop):
            return True
    return False

def _rebuild_loop(loops:list[Loop]):
    all_edges=sum([list(loop.edges) for loop in loops],[])
    rebuilt_loops=FindLoopAlgo(all_edges,directed=True,cancel_out_opposite=False).get_result()
    return rebuilt_loops

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,REBUILD_AND_CANCEL_LOOP[1]+1)],
    ids=[f"case_{i}" for i in range(1,REBUILD_AND_CANCEL_LOOP[1]+1)],
)
def test_rebuild_and_cancel_loop(case)->None:
    """重建Loop有向图拓扑，去除反向边"""
    inputs=read_case(REBUILD_AND_CANCEL_LOOP,case["in"],hook_mode="cad")
    loops=[g for g in inputs if isinstance(g,Loop)]
    res=_rebuild_and_cancel_loop(loops)
    if __name__=="__main__": 
        CADPlotter.draw_geoms(res)
        # write_stdout(res,REBUILD_AND_CANCEL_LOOP,f"out_{i}")     
    else:   
        std_out=read_case(REBUILD_AND_CANCEL_LOOP,case["out"])
        assert len(res)==len(std_out)
        std_tree=STRTree(std_out)
        assert all([_find_identical_loop(geom,std_tree) for geom in res])

def _rebuild_and_cancel_loop(loops:list[Loop]):
    all_edges=sum([list(loop.edges) for loop in loops],[])
    rebuilt_loops=FindLoopAlgo(all_edges,directed=True,cancel_out_opposite=True).get_result()
    return rebuilt_loops

# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"out_{i}"} for i in range(1,FIND_LOOP[1]+1)],
    ids=[f"case_{i}" for i in range(1,FIND_LOOP[1]+1)],
)
def test_find_loop(case):
    """用边集重建无向图拓扑（找环）"""
    inputs=read_case(FIND_LOOP,case["in"],hook_mode="cad")
    polys=[g for g in inputs if isinstance(g,(Polyedge,Loop))]
    edges=[g for g in inputs if isinstance(g,Edge)]
    for g in polys:
        edges.extend(g.edges)            
    res=FindLoopAlgo(edges).get_result()
    if __name__=="__main__":
        # write_stdout(res,FIND_LOOP,f"out_{i}") 
        CADPlotter.draw_geoms(res)
    else:
        std_out=read_case(FIND_LOOP,case["out"])
        assert len(res)==len(std_out)
        std_tree=STRTree(std_out)
        assert all([_find_identical_loop(geom,std_tree) for geom in res])

# -------------------------------------------------------------------------

if __name__=="__main__":
    if 0:
        for i in range(1,COVER_NODE[1]+1):
            test_loop_covers_node({"in":f"case_{i}","out":f"out_{i}"})
    if 0:
        for i in range(1,COVER_EDGE[1]+1):
            test_loop_covers_edge({"in":f"case_{i}","out":f"out_{i}"})
    if 0:
        for i in range(1,COVER_LOOP[1]+1):
            test_loop_covers_loop({"in":f"case_{i}","out":f"out_{i}"})
    if 0:
        for i in range(1,REBUILD_LOOP[1]+1):
            test_rebuild_loop({"in":f"case_{i}","out":f"out_{i}"})
    if 0:
        for i in [8]:
        # for i in range(1,REBUILD_AND_CANCEL_LOOP[1]+1):
            test_rebuild_and_cancel_loop({"in":f"case_{i}","out":f"out_{i}"})
    if 0:
        for i in [7]:
        # for i in range(1,FIND_LOOP[1]+1):
            test_find_loop({"in":f"case_{i}","out":f"out_{i}"})
    ...