import json
import matplotlib.pyplot as plt
from lib.geom import Node,Edge,Loop,Poly,_draw_polygon
from lib.linalg import Vec3d
from lib.utils import Timer, Constant as const
from shapely import Polygon,MultiPolygon,Point
from shapely.ops import nearest_points
from lib.room_algo import make_cover_tree

def read_data(fname:str)->tuple[list[Loop],list[Edge]]:
    with open(fname,'r') as f:
        polylines=json.load(f)
    loops,doors=[],[]
    for polyline in polylines:
        if polyline["layer"]=="WALL":
            nodes=[Node(coord['x'],coord['y']) for coord in polyline["coords"]]
            loops.append(Loop.from_nodes(nodes))
            loops[-1].simplify()
            if loops[-1].area<0:
                loops[-1].reverse()
        elif polyline["layer"]=="DOOR":
            s=Node(polyline["coords"][0]['x'],polyline["coords"][0]['y'])
            e=Node(polyline["coords"][1]['x'],polyline["coords"][1]['y'])
            doors.append(Edge(s,e))
    return loops,doors
def get_pocket_rect(door:Edge,wall:Edge,WIDTH_LIMIT:float)->tuple[list[Node],Node]:
    """计算门后面的口袋矩形"""
    door_norm=door.to_vec3d().unit()
    wall_norm=wall.to_vec3d().unit()
    if door_norm.dot(wall_norm)<0:
        door.reverse()
    other_side=door.offset(-WIDTH_LIMIT) #口袋的底边
    points=[None]*4
    # 口袋的范围：从门边开始算，不超过所在的墙端点
    covered_length=max(WIDTH_LIMIT,door.length) #计算小球需要覆盖的长度
    length_e2s=min(covered_length,door.e.dist(wall.s)-1) #从门起点到墙终点取距离
    length_s2e=min(covered_length,door.s.dist(wall.e)-1) #从门终点到墙起点取距离
    points[0]=door.e.to_vec3d()-wall_norm*length_e2s
    points[1]=other_side.e.to_vec3d()-wall_norm*length_e2s
    points[2]=other_side.s.to_vec3d()+wall_norm*length_s2e
    points[3]=door.s.to_vec3d()+wall_norm*length_s2e
    ave=sum(points,start=Vec3d(0,0))/4
    pocket_center=Node.from_vec3d(ave)
    return [Node(p.x,p.y) for p in points],pocket_center
if __name__=="__main__":
    plt.figure()
    WIDTH_LIMIT=1500 #校验宽度
    for case_num in range(1,21):
        plt.subplot(4,6,case_num)
        ax = plt.gca()
        ax.set_aspect(1)
        # 读墙门数据
        loops,doors=read_data(f"tool\\find_passage\\test_cases\\case{case_num}.json")
        doors_on_loop={loop:[] for loop in loops} 
        for door in doors:
            for loop in loops:
                for edge in loop.edges:
                    if door.point_at(0.5)[0].is_on_edge(edge):
                        doors_on_loop[loop].append(door)
        
        # loop组成房间polygon
        cover_tree=make_cover_tree(loops)
        for tree_node in cover_tree:
            if tree_node.parent is None:
                for ch in tree_node.child:
                    ch.obj.reverse()
                break
        
        # 在每个门的地方插一个矩形口袋
        pocket_centers={}
        for i in range(len(loops)):
            for door in doors_on_loop[loops[i]]:
                loop=loops[i]
                for j in range(len(loop.edges)):
                    edge=loop.edges[j]
                    if door.point_at(0.5)[0].on_edge(edge): #门在墙线上
                        pocket_rect,pocket_center=get_pocket_rect(door,edge,WIDTH_LIMIT)
                        pocket_centers[door]=pocket_center
                        plt.scatter(pocket_center.x,pocket_center.y,c='r') #画门路径起点
                        loop_nodes=loop.nodes()
                        for k in range(4):
                            plt.scatter(pocket_rect[k].x,pocket_rect[k].y,c='b') #画口袋
                            loop_nodes.insert(j+k+1,pocket_rect[k])
                        loops[i]=Loop.from_nodes(loop_nodes)
                        doors_on_loop[loops[i]]=doors_on_loop[loop]
                        break
                    else: continue
                    break

        # 插完口袋的房间轮廓
        for loop in loops:
            plt.plot(*loop.xy,color='k')
        
        # 向内offset，生成路径范围polygon
        loops_offset=[]
        holes=[]
        for loop in loops:
            if loop.area>0:
                dist=-(WIDTH_LIMIT/2-2)
                shell=loop.polygon.buffer(dist)
            else:
                dist=WIDTH_LIMIT/2-2
                holes.append(loop.polygon.buffer(dist))
        for hole in holes:
            shell=shell.difference(hole)
        loops_offset=list(shell.geoms) if isinstance(shell,MultiPolygon) else [shell]
        for loop in loops_offset:
            _draw_polygon(loop,color=('g','g'))

        # 判断门所属的路径范围
        poly_dict={}
        for door in doors:
            for loop in loops_offset:
                pair=nearest_points(Point(pocket_centers[door].x,pocket_centers[door].y),loop)
                if pair[0].distance(pair[1])<2:
                    poly_dict[door]=loop
                    break

        # 任意两个待验证的门所属的路径范围不一致，表示无法联通
            for i in range(len(doors)-1):
                for j in range(i+1,len(doors)):
                    if doors[i] in poly_dict and doors[j] in poly_dict and \
                        poly_dict[doors[i]] is not poly_dict[doors[j]]:
                        # 无法联通的门之间画虚线
                        plt.plot([doors[i].point_at(0.5)[0].x,doors[j].point_at(0.5)[0].x],
                                [doors[i].point_at(0.5)[0].y,doors[j].point_at(0.5)[0].y],
                                'r--') 

    plt.show()
