import pandas as pd
import matplotlib.pyplot as plt
from lib.geom import Node,Edge,Loop,Polygon,GeomUtil
from lib.index import TreeNode
from lib.utils import Timer, Constant
from lib.geom import Edge,Arc,LineSeg,Loop
from lib.building_element import Wall
from lib.geom_algo import GeomAlgo
from itertools import groupby


class MergeWallAlgo(GeomAlgo):
    def __init__(self,walls:list[Wall],const:Constant=None) -> None:
        """合并平行且有重叠的墙

        Args:
            walls (list[Wall]): 任意一组待合并的墙.
            const (Constant, optional): 误差控制常量. Defaults to None.
        """
        super().__init__(const=const)
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
        parallel_groups.extend()
        # 直墙分组
        line_walls=filter(lambda wall:isinstance(wall.base,LineSeg),walls)
        walls.sort(key=lambda wall:wall.base.angle)
        current_angle=-self.const._MAX_VAL
        for line in line_walls:
            if line.angle-current_angle>self.const.TOL_ANG: # !parallel
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
    def __init__(self,edges:list[Wall],const:Constant=None) -> None:
        self.const=const or Constant.default()
        self.edges:list[Edge]=edges
        self.loops:list[Loop]=[]
    def _preprocess(self) -> None:
        super()._preprocess()
    def _postprocess(self) -> None:
        super()._postprocess()

# %% 测试
if 1 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt

    const=Constant.default()
    with open("test/find_room/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)

    nodes=[]
    for obj in j_obj:
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
                s=GeomUtil.find_or_insert_node(s,nodes)
                e=GeomUtil.find_or_insert_node(e,nodes)
                s.add_edge_in_order(Edge(s,e,lw,rw))
                e.add_edge_in_order(Edge(e,s,rw,lw))

    # 找环
    loops=find_loop(nodes)

    # 按墙厚offset
    offset_loops:list[Loop] = []
    for loop in loops:
        offset_loops+=loop.offset(side="left",split=True,mitre_limit=20000)
    for loop in offset_loops:
        plt.plot(*loop.xy)

    # loop组成房间polygon
    cover_tree=make_cover_tree(offset_loops)
    rooms:list[Polygon]=[]
    for tree_node in cover_tree:
        if tree_node.obj.area>0:
            new_shell=tree_node.obj
            new_holes=[ch.obj for ch in tree_node.child]
            rooms.append(Polygon(new_shell,new_holes))

    with Timer(tag="画图"):
        # 画墙基线
        for loop in loops:
            for edge in loop.edges:
                other=edge.point_at(0.3)
                plt.plot([edge.s.x,other.x],[edge.s.y,other.y],color="m")
        # 画房间
        for room in rooms:
            _draw_polygon(room.polygon,color=('y','g'))
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
