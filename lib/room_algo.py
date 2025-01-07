import pandas as pd
import matplotlib.pyplot as plt
from lib.geom import Node,Edge,Loop,Polygon,GeomUtil
from lib.index import TreeNode
from lib.utils import Timer, Constant



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
