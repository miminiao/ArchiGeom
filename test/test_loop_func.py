"""测试环相关的操作"""
import pytest

import json
from lib.geom import Geom,Node,LineSeg,Arc,Edge,Polyedge,Loop,Polygon
from lib.geom_algo import BooleanOperation
from lib.geom_plotter import MPLPlotter,CADPlotter
from tool.converter.json_converter import JsonLoader
from lib.utils import Timer

DIR_PATH="./test/loop_func/"
COVER_NODE=("loop_covers_node/",8)
LOOP_UNION=("loop_union/",1)

def read_input_geoms(path)->list[Geom]:
    with open(path,'r') as f:
        geoms=json.load(f,object_hook=JsonLoader.from_cad_obj)
    return geoms
def read_std_out(path):
    with open(path,'r') as f:
        return json.load(f)
    
@pytest.mark.parametrize(
    argnames="case",
    argvalues=[{"in":f"case_{i}","out":f"case_{i}_out"} for i in range(1,COVER_NODE[1])],
    ids=[f"case_{i}" for i in range(1,COVER_NODE[1])],
)
def test_loop_covers_node(case):
    geoms=read_input_geoms(DIR_PATH+COVER_NODE[0]+case["in"])
    std_out=read_std_out(DIR_PATH+COVER_NODE[0]+case["out"])
    covered_idx=loop_covers_node(geoms,draw=False)

    assert len(covered_idx)==len(std_out)
    assert all([covered_idx[i]==std_out[i] for i in range(len(std_out))])

def loop_covers_node(geoms:list[Geom],draw:bool=False)->None:
    loop=[g for g in geoms if isinstance(g,(Polyedge,Loop))][0]
    points=[g.center for g in geoms if isinstance(g,Arc) and g.angles[0]==0]
    covered,others=[],[]
    for p in points:
        if loop._covers_node(p): 
            covered.append(p)
        else: others.append(p)
    if draw:
        CADPlotter.draw_geoms([loop])
        CADPlotter.draw_geoms(covered,color=3)
        CADPlotter.draw_geoms(others,color=1)
    else:
        return [geoms.index()]
@Timer()
def test_loop_covers_edge(geoms:list[Geom])->None:
    loop=[g for g in geoms if isinstance(g,Loop)][0]
    CADPlotter.draw_geoms([loop])
    edges=[g for g in geoms if isinstance(g,Edge)]
    covered,others=[],[]
    for p in edges:
        if loop._covers_edge(p): 
            covered.append(p)
        else: others.append(p)
    CADPlotter.draw_geoms(covered,color=3)
    CADPlotter.draw_geoms(others,color=1)
@Timer()
def test_loop_covers_loop(geoms:list[Geom])->None:
    loops=[g for g in geoms if isinstance(g,Loop)]
    covered=[False,False]
    if loops[0]._covers_polyedge(loops[1]):
        covered[1]=True
    if loops[1]._covers_polyedge(loops[0]):
        covered[0]=True
    CADPlotter.draw_geoms([loops[0]],color=3 if covered[0] else 1)        
    CADPlotter.draw_geoms([loops[1]],color=3 if covered[1] else 1)

def loop_union(loops:list[Loop],draw:bool=False)->list[Polygon]:
    return BooleanOperation._loop_union(loops,const=Geom.const)

if __name__=="__main__":
    # for i in range(1,8):
    #     geoms=read_input_geoms(DIR_PATH+COVER_NODE[0]+"case_{i}.json")
    #     loop_covers_node(geoms,draw=True)
    # for i in range(1,8):
    #     geoms=read_input_geoms(f"test/loop_func/loop_covers_edge/case_{i}.json")
    #     test_loop_covers_edge(geoms)
    # for i in range(1,14):
    #     geoms=read_input_geoms(f"test/loop_func/loop_covers_loop/case_{i}.json")
    #     test_loop_covers_loop(geoms)
    for i in range(1,LOOP_UNION[1]):
        geoms=read_input_geoms(DIR_PATH+LOOP_UNION[0]+"case_{i}.json")
        loop_union(geoms,draw=True)    