import json
from lib.geom import Geom,Node,Edge,Polyedge,Loop,Polygon
from lib.geom_algo import FindLoopAlgo
from lib.geom_plotter import MPLPlotterr,CADPlotter
from tool.converter.json_converter import JsonLoader

def read_geoms()->list[Geom]:
    with open("test/find_loop/case_1.json",'r') as f:
        geoms=json.load(f,object_hook=JsonLoader.from_cad_obj)
    polys=[g for g in geoms if isinstance(g,(Polyedge,Loop))]
    edges=[g for g in geoms if isinstance(g,Edge)]
    for g in polys:
        edges.extend(g.edges)
    return edges

edges=read_geoms()
loops=FindLoopAlgo(edges).get_result()

# MPLPlotterr.draw_geoms(geoms,show=True)
CADPlotter.draw_geoms(loops)