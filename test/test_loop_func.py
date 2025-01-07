import json
from lib.geom import Geom,Node,LineSeg,Arc,Edge,Polyedge,Loop
from lib.geom_plotter import MPLPlotter,CADPlotter
from tool.converter.json_converter import JsonLoader
from lib.utils import Timer
def read_geoms(path)->list[Geom]:
    with open(path,'r') as f:
        geoms=json.load(f,object_hook=JsonLoader.from_cad_obj)
    return geoms
@Timer()
def test_loop_covers_node(geoms:list[Geom])->None:
    loop=[g for g in geoms if isinstance(g,(Polyedge,Loop))][0]
    CADPlotter.draw_geoms([loop])
    points=[g.center for g in geoms if isinstance(g,Arc) and g.angles[0]==0]
    covered,others=[],[]
    for p in points:
        if loop._covers_node(p): 
            covered.append(p)
        else: others.append(p)
    CADPlotter.draw_geoms(covered,color=3)
    CADPlotter.draw_geoms(others,color=1)
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
for i in range(1,8):
    geoms=read_geoms(f"test/loop_func/loop_covers_node/case_{i}.json")
    test_loop_covers_node(geoms)
for i in range(1,8):
    geoms=read_geoms(f"test/loop_func/loop_covers_edge/case_{i}.json")
    test_loop_covers_edge(geoms)
for i in range(1,14):
    geoms=read_geoms(f"test/loop_func/loop_covers_loop/case_{i}.json")
    test_loop_covers_loop(geoms)