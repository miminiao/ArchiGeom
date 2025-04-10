import math
from lib.geom import Node,Edge,Arc,LineSeg,Loop,Polygon,GeomUtil
from lib.utils import Timer, Constant as Const
from lib.building_element import Wall
from lib.geom_algo import GeomAlgo,FindLoopAlgo
from itertools import groupby


class MergeWallAlgo(GeomAlgo):
    def __init__(self,walls:list[Wall]) -> None:
        """合并平行且有重叠的墙

        Args:
            walls (list[Wall]): 任意一组待合并的墙.
        """
        super().__init__()
        self.walls=walls
    def _preprocess(self)->None:
        super()._preprocess()
    def _postprocess(self)->None:
        super()._postprocess()        
    def _get_parallel_groups(self,walls:list[Wall])->list[list[Wall]]:
        """按平行分组"""
        parallel_groups=[]
        # 圆弧墙分组
        arc_walls=filter(lambda wall:isinstance(wall.base,Arc),walls)
        arc_groups=groupby(arc_walls,key=lambda wall:wall.base.center)
        parallel_groups.extend(arc_groups)
        # 直墙分组
        line_walls=filter(lambda wall:isinstance(wall.base,LineSeg),walls)
        walls.sort(key=lambda wall:wall.base.angle)
        current_angle=-math.inf
        for line in line_walls:
            if line.angle-current_angle>Const.TOL_ANG: # !parallel
                new_group=[line]
                parallel_groups.append(new_group)
                current_angle=line.angle
            else: 
                new_group.append(line)
        return parallel_groups
    def get_result(self):
        for i in range(len(self.walls)-1):
            for j in range(i+1,len(self.walls)):
                if self.walls[i].base:
                    pass
class FindRoomAlgo(GeomAlgo): #TODO
    def __init__(self,edges:list[Wall]) -> None:
        self.edges:list[Edge]=edges
        self.loops:list[Loop]=[]
    def _preprocess(self) -> None:
        super()._preprocess()
    def _postprocess(self) -> None:
        super()._postprocess()

# %% 测试
if __name__ == "__main__":
    import json
    from tool.converter.json_converter import JsonLoader
    from lib.geom_algo import BooleanOperation
    from lib.geom_plotter import MPLPlotter
    import matplotlib.pyplot as plt

    with open("test/find_room/case_1.json",'r',encoding="utf8") as f:
        walls=json.load(f,object_hook=JsonLoader.from_cad_obj)

    # 找环
    loops=FindLoopAlgo(walls).get_result()

    # 按墙厚offset
    offset_loops:list[Loop] = []
    for loop in loops:
        offset_loops+=loop.offset(side="left",split=True,mitre_limit=20000)
    for loop in offset_loops:
        plt.plot(*loop.xy)

    # loop组成房间polygon
    rooms=BooleanOperation._rebuild_loop_topology(offset_loops)

    with Timer(tag="画图"):
        # 画墙基线
        for loop in loops:
            for edge in loop.edges:
                other=edge.point_at(0.3)
                plt.plot([edge.s.x,other.x],[edge.s.y,other.y],color="m")
        # 画房间
        for room in rooms:
            MPLPlotter.draw_geoms(room.polygon,color=('y','g'))
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
