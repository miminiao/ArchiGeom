import random
import typing
import math
import matplotlib.pyplot as plt
from lib.geom import Node,LineSeg,Arc,Loop,Polygon
from lib.geom_algo import BreakEdgeAlgo, MergeEdgeAlgo,FindOutlineAlgo,FindConnectedGraphAlgo
from lib.building_element import Wall,Window,Door
from lib.interval import Interval1d,MultiInterval1d
from lib.utils import Timer, Constant
from lib.linalg import Vec3d
from lib.index import STRTree
from typing import Union,Callable


class Wall(LineSeg):
    def __init__(self, s: Node, e: Node, lw=0, rw=0) -> None:
        super().__init__(s, e, lw, rw)
        self.interval:Union[Interval1d,MultiInterval1d]=None
        self.upper_interval:Union[Interval1d,MultiInterval1d]=None
        self.lower_interval:Union[Interval1d,MultiInterval1d]=None
        self.upper_lines:list[Wall]=[]
        self.lower_lines:list[Wall]=[]
        self.linked_lines:list[LineSeg]=[]
def find_or_insert_node(target_node:Node,nodes:list[Node])->Node:
    for node in nodes:
        if target_node.equals(node):
            return node
    nodes.append(target_node)
    return target_node
def parse_data(j_data:list,center_to_zero:bool=True)->tuple:
    def map_layer(layer:str)->list:
        if "WALL"==layer.upper(): return wall_lines
        if "AE-FLOR"==layer.upper(): return watch_lines
        if "AE-WIND"==layer.upper(): return watch_lines
        if "看线" in layer: return watch_lines
        if "A-BLCN" in layer.upper(): return watch_lines
    def map_block_name(block_name:str)->list:
        if "$DORLIB2D$" in ent["block_name"].upper(): return door_blocks
        if "DOOR2D" in ent["block_name"].upper(): return door_blocks
        if "WIN2D" in ent["block_name"].upper(): return window_blocks
        if "$WINLIB2D$" in ent["block_name"].upper(): return window_blocks
    nodes:list[Node]=[]
    wall_lines:list[LineSeg]=[]
    watch_lines:list[LineSeg]=[]
    opening_lines:list[LineSeg]=[]
    window_blocks=[]
    door_blocks=[]
    block_defs={}
    trans=None  # 归零向量
    for ent in j_data:
        match ent["object_name"]:
            case "block_ref":
                category=map_block_name(ent["block_name"])
                if category is not None: category.append(ent)
            case "block_def":
                block_defs[ent["block_name"]]=ent
            case "line":
                start_vec=Vec3d(*ent["start_point"])
                end_vec=Vec3d(*ent["end_point"])
                trans=trans or (-start_vec)  # 归零向量
                s=Node.from_vec3d(start_vec+trans)
                e=Node.from_vec3d(end_vec+trans)
                if s.equals(e):continue
                s=find_or_insert_node(s,nodes)
                e=find_or_insert_node(e,nodes)
                s.add_edge_in_order(LineSeg(s,e))
                e.add_edge_in_order(LineSeg(e,s))
                new_edge=LineSeg(s,e)
                category=map_layer(ent["layer"])
                if category is not None: category.append(new_edge)
            case "arc":
                center_vec=Vec3d(*ent["center"])
                trans=trans or -center_vec  # 归零向量
                c=Node.from_vec3d(center_vec+trans)
                r=ent["radius"]
                normal=ent["normal"]
                start_angle,end_angle=ent["start_angle"],ent["end_angle"]
                s_vec,e_vec=Vec3d(*ent["start_point"]),Vec3d(*ent["end_point"])
                start_point=Node.from_vec3d(s_vec+trans)
                end_point=Node.from_vec3d(e_vec+trans)
                tot_angle=end_angle-start_angle
                if tot_angle<0: tot_angle+=math.pi*2
                new_edge=Arc(start_point,end_point,math.tan(tot_angle/4))
                if new_edge.length<const.TOL_DIST:continue
                category=map_layer(ent["layer"])
                if category is not None: 
                    # category.append(new_LineSeg)  # TODO
                    category.append(LineSeg(new_edge.s,new_edge.e))  # 先按直线段处理
            case "polyline":
                seg_num=len(ent["segments"]) if ent["is_closed"] else len(ent["segments"])-1
                for i in range(seg_num):
                    seg=ent["segments"][i]
                    x1,y1,_=seg["start_point"]
                    x2,y2,_=ent["segments"][(i+1)%len(ent["segments"])]["start_point"]
                    bulge=seg["bulge"]
                    if trans is None: trans=(-x1,-y1)  # 归零向量
                    s=Node(x1+trans[0],y1+trans[1])
                    e=Node(x2+trans[0],y2+trans[1])
                    if s.equals(e):continue
                    s=find_or_insert_node(s,nodes)
                    e=find_or_insert_node(e,nodes)
                    s.add_edge_in_order(LineSeg(s,e))
                    e.add_edge_in_order(LineSeg(e,s))
                    new_edge=LineSeg(s,e) if abs(bulge)<const.TOL_VAL else Arc(s,e,bulge)
                    category=map_layer(ent["layer"])
                    if category is not None: category.append(new_edge)
            case _:continue
    return wall_lines,watch_lines,opening_lines,window_blocks,door_blocks,block_defs
def group_lines_by_angle(lines:list[LineSeg])->tuple[list[list[LineSeg]],list[Vec3d],list[Vec3d]]:
    """将平行的墙线分成一组，并按法向量排序

    Args:
        lines (list[LineSeg]): 平行线.

    Returns:
        list[list[LineSeg]]: 分组并排序的平行线.
        list[Vec3d]: 单位向量.
        list[Vec3d]: 法向量(右).
    """
    groups:list[list[LineSeg]]=[]
    unit_vectors:list[Vec3d]=[]
    normal_vectors:list[Vec3d]=[]
    lines.sort(key=lambda line:line.angle_of_line)
    current_angle=-const._MAX_VAL
    for line in lines:
        if line.angle_of_line-current_angle>const.TOL_ANG:
            new_group=[line]
            groups.append(new_group)
            current_angle=line.angle_of_line
        else:
            new_group.append(line)
    # 排序
    for line_group in groups:
        unit_vector,normal_vector=get_unit_and_normal_of_parallel_lines(line_group)
        line_group.sort(key=lambda line:line.s.to_vec3d().dot(normal_vector))
        unit_vectors.append(unit_vector)
        normal_vectors.append(normal_vector)
    return groups,unit_vectors,normal_vectors
def get_unit_and_normal_of_parallel_lines(lines:list[LineSeg])->tuple[Vec3d,Vec3d]:
    """返回一组平行线的单位向量和法向量"""
    # 找一根最长的，作为方向向量
    longest_line=max(lines,key=lambda line:line.length)
    # 沿法向量（右转90度）排序
    unit_vector=longest_line.to_vec3d().unit()  # 单位向量
    normal_vector=unit_vector.cross(Vec3d(0,0,1))  # 法向量
    return unit_vector,normal_vector
def find_walls_from_parallel_lines(line_group:list[LineSeg],wall_width_limit:float)->list[Wall]:
    """将一组平行线转化为一组墙，仅识别宽度<=wall_width_limit的墙"""
    unit_vector,normal_vector=get_unit_and_normal_of_parallel_lines(line_group)
    # 查找所有与当前line_i距离<=dist_limit且有重叠部分的平行线line_j，有多层重叠时取距离最近的一段组成墙，墙中线=重叠部分的中线
    walls:list[Wall]=[]
    for line in line_group:
        line.upper_interval=Interval1d(line.s.to_vec3d().dot(unit_vector),line.e.to_vec3d().dot(unit_vector),const._MAX_VAL)
        line.lower_interval=line.upper_interval.copy()
    for i,line_i in enumerate(line_group):
        dist_i=line_i.s.to_vec3d().dot(normal_vector)
        for j in range(i+1,len(line_group)):
            line_j=line_group[j]
            dist_j=line_j.s.to_vec3d().dot(normal_vector)
            dist=dist_j-dist_i
            if dist>wall_width_limit[1]+const.TOL_DIST:
                break
            if line_i.upper_interval.is_overlap(line_j.upper_interval):
                # line_i.upper_lines.append(line_j)
                # line_j.lower_lines.append(line_i)
                intersection=line_i.upper_interval*line_j.upper_interval
                line_i.upper_interval+=intersection.copy(h=dist)
    walls=[]
    for line_i in line_group:
        walls+=get_walls_from_interval(line_i)

    # 处理并排的墙
    for i in range(len(walls)-1):
        wall_i=walls[i]
        dist_i=wall_i.s.to_vec3d().dot(normal_vector)
        for j in range(i+1,len(walls)):
            wall_j=walls[j]
            dist_j=wall_j.s.to_vec3d().dot(normal_vector)
            dist=dist_j-dist_i
            if dist>wall_width_limit[1]+const.TOL_DIST: break
            if (abs(wall_i.lw+wall_j.lw-dist)<const.TOL_DIST
                and wall_i.interval.is_overlap(wall_j.interval)):
                intersection=wall_i.interval*wall_j.interval
                wall_i.upper_interval+=intersection
                wall_j.lower_interval+=intersection
    for i in range(1,len(walls)-1):
        wall_i=walls[i]
        if wall_i.upper_interval.is_overlap(wall_i.lower_interval):
            intersection=wall_i.upper_interval*wall_j.lower_interval
            wall_i.interval-=intersection
    return walls

def get_walls_from_interval(line:Wall)->list[Wall]:
    """根据重叠区间求墙基线"""
    walls=[]
    if isinstance(line.upper_interval,Interval1d):
        line.upper_interval=MultiInterval1d([line.upper_interval])
    unit_vector=line.to_vec3d().unit()
    for sub_intv in line.upper_interval._items:
        if sub_intv.value==const._MAX_VAL or sub_intv.value<const.TOL_DIST: continue
        s_pos=line.s.to_vec3d().dot(unit_vector)
        s_vector=line.s.to_vec3d()+unit_vector*(sub_intv.l-s_pos)
        e_vector=line.s.to_vec3d()+unit_vector*(sub_intv.r-s_pos)
        new_baseline=LineSeg(Node(s_vector.x,s_vector.y),Node(e_vector.x,e_vector.y)).offset(sub_intv.value/2)
        new_wall=Wall(new_baseline.s,new_baseline.e,lw=sub_intv.value/2,rw=sub_intv.value/2)
        new_wall.upper_interval=sub_intv.copy(value=const._MAX_VAL)
        new_wall.lower_interval=sub_intv.copy(value=const._MAX_VAL)
        new_wall.interval=sub_intv.copy(value=0)
        new_wall.linked_lines=[]
        for l in new_wall.linked_lines:
            l.linked_walls.append(new_wall)
        walls.append(new_wall)
    return walls
def remove_watch_window_and_railing(watch_lines:list[LineSeg],
                                    wall_width_limit:tuple[float,float],
                                    rail_width_limit:tuple[float,float],
                                    )->list[LineSeg]:
    """识别墙宽范围内的>=3根的平行线，当作窗和栏杆先清除掉，只保留最边上的两根"""
    groups=group_lines_by_angle(watch_lines)
    removed_lines=set()  # 记录需要删除的
    for lines,unit_vector,normal_vector in zip(*groups):
        intervals={line:Interval1d(line.s.to_vec3d().dot(unit_vector),line.e.to_vec3d().dot(unit_vector),0) for line in lines}
        for head in range(len(lines)-2):
            intv_head=intervals[lines[head]]
            for tail in range(head+2,len(lines)):
                # 如果平行距离>limit，就不继续增加tail了
                if lines[tail].s.to_vec3d().dot(normal_vector)-lines[head].s.to_vec3d().dot(normal_vector)>wall_width_limit[1]+const.TOL_DIST: 
                    break
                # 如果不重叠，就不枚举中间的线了
                intv_tail=intervals[lines[tail]]
                if (intv_head*intv_tail).is_empty(): 
                    continue
                # 枚举中间的线，如果和head&tail都重叠，就标记为删除
                for i in range(head+1,tail):
                    intv_i=intervals[lines[i]]
                    if not (intv_i*intv_head).is_empty() and not (intv_i*intv_tail).is_empty():
                        removed_lines.add(lines[i])
    return list(filter(lambda line:line not in removed_lines,watch_lines)), list(removed_lines)
def find_butress(wall_outline_loops:list[Loop])->set[LineSeg]:
    """根据墙线轮廓，找到墙垛"""
    butresses=set()
    for outline in wall_outline_loops:
        # 找到2次90度左转的墙线，作为备选墙垛
        for i,edge in enumerate(outline.edges):
            if not WALL_WIDTH_LIMITS[0]-const.TOL_DIST<=edge.length<=WALL_WIDTH_LIMITS[1]+const.TOL_DIST:
                continue
            pre_edge=outline.edges[i-1]
            succ_edge=outline.edges[(i+1)%len(outline.edges)]
            if (abs(pre_edge.to_vec3d().unit().cross(edge.to_vec3d().unit()).z-1)<const.TOL_VAL
                and abs(edge.to_vec3d().unit().cross(succ_edge.to_vec3d().unit()).z-1)<const.TOL_VAL): # 2次90度左转
                butresses.add(edge)
        # 连续3个墙垛相邻的情况，去除夹在中间的那个
        for i,edge in enumerate(outline.edges):
            pre_edge=outline.edges[i-1]
            succ_edge=outline.edges[(i+1)%len(outline.edges)]
            if edge in butresses and pre_edge in butresses and succ_edge in butresses: # 夹在中间
                if len(outline.edges)==4 and (edge.length<pre_edge.length or edge.length<succ_edge.length): # 连续4个的情况，去除长的； TODO：都一样长的情况，根据配对关系判断 
                    continue
                butresses.remove(edge)
        # 连续2个墙垛相邻的情况：比较当前墙垛与另一墙垛的邻边的长度，如果比邻边，把与长的相邻的从墙垛中踢掉
        for i,edge in enumerate(outline.edges):
            pre_edge=outline.edges[i-1]
            if edge in butresses and pre_edge in butresses:  # 连续2个墙垛
                succ_neighbor=outline.edges[(i+1)%len(outline.edges)]  # 下一条邻边
                pre_neighbor=outline.edges[i-2]  # 上一条邻边
                if pre_neighbor.length<edge.length:
                    butresses.remove(edge)
                elif succ_neighbor.length<pre_edge.length:
                    butresses.remove(pre_edge)
    return butresses
def find_opening_regions(butresses:set[LineSeg],
                         wall_outline_loops:list[Loop],
                         butress_dist_limits:tuple[float,float],
                         )->list[Loop]:
    """根据墙垛，找到墙洞区域

    Args:
        butresses (set[LineSeg]): _description_
        wall_outline_loops (list[Loop]): _description_
        butress_dist_limits (tuple[float,float]): _description_

    Returns:
        list[Loop]: Loop.edges=[s1,e1,e2,s2]
    """
    def match_butress_lines(dist_condition:Callable[[LineSeg,LineSeg],bool]):
        for i,line in enumerate(lines):
            """从一组平行线中找到墙垛所形成的墙洞区域"""
            # 遍历每一个墙垛
            if line not in butresses or line in visited_butresses: continue
            # 在平行的墙线中，找到与当前墙垛方向相对的最近的一段，配对
            if line.to_vec3d().unit().dot(unit_vector)>0:
                others_idx=range(i+1,len(lines))
            else:
                others_idx=range(i-1,-1,-1)
            for j in others_idx:
                other=lines[j]
                if (intervals[line].is_overlap(intervals[other], include_endpoints=False)  # 有重叠部分
                    and other.to_vec3d().unit().dot(line.to_vec3d().unit())<0  # 且方向相对
                    and other.projection(line.s).dist(line.s)>const.TOL_DIST  # 且不共线（防止面域没闭合的情况）
                ):
                    if dist_condition(line,other):  # 如果满足距离条件，就直接配对
                        visited_butresses.add(other)
                        visited_butresses.add(line)
                        opening_regions.append(Loop.from_nodes([line.s, line.e, other.projection(line.e),other.projection(line.s)]))
                    break  # 只找最近的一个判断
        return
    def is_pair_butress(line:LineSeg,other:LineSeg)->bool:
        """两个相对的垛子"""
        d=other.projection(line.s).dist(line.s)
        return (other in butresses  # 1.对面也是个垛子
                and d<=butress_dist_limits[0]+const.TOL_DIST  # 2.平行距离满足条件
                )
    def is_single_butress(line:LineSeg,other:LineSeg)->bool:
        """一个垛子和一个墙"""
        neighbors=neighbor_butress[line]
        if len(neighbors)==2:  # 1.当前的垛子如果夹在两个垛子中间，就忽略
            return False 
        if len(neighbors)==1:
            if neighbors[0] in visited_butresses:  # 2.当前垛子相邻的边也是个垛子，且已经配对了，就忽略
                return False 
            if neighbors[0].length<line.length:  # 3.当前垛子相邻的边也是个垛子，且长度比当前的要短，就忽略
                return False
        d=other.projection(line.s).dist(line.s)  # 平行距离
        return (other in wall_lines  # 1.对面是个墙
                and d<=butress_dist_limits[1]+const.TOL_DIST  # 2.平行距离满足条件
                )
    # 记录每个墙垛的相邻墙垛
    neighbor_butress:dict[LineSeg,list[LineSeg]]={}
    for loop in wall_outline_loops:
        for i,edge in enumerate(loop.edges):
            if edge in butresses:
                neighbor_butress[edge]=[]
                if loop.edges[i-1] in butresses:
                    neighbor_butress[edge].append(loop.edges[i-1])
                if loop.edges[(i+1)%len(loop.edges)] in butresses:
                    neighbor_butress[edge].append(loop.edges[(i+1)%len(loop.edges)])
    # 沿着墙垛的方向，找到最近的有重叠范围的平行线
    wall_lines=[edge for loop in wall_outline_loops for edge in loop.edges]
    opening_regions=[]
    visited_butresses=set()
    groups=group_lines_by_angle(wall_lines)
    # groups=group_lines_by_angle(all_lines)
    for lines,unit_vector,normal_vector in zip(*groups):
        # 初始化interval
        intervals:dict[LineSeg,Interval1d]={}
        for line in lines:
            l=line.s.to_vec3d().dot(unit_vector)
            r=line.e.to_vec3d().dot(unit_vector)
            intervals[line]=Interval1d(l,r,0) if r>l else Interval1d(r,l,0)
        # 先找成对的垛子
        match_butress_lines(is_pair_butress)
        # 再找单个的垛子
        match_butress_lines(is_single_butress)
    # 去除互相重叠的洞口区域 TODO
    return opening_regions
def recognize_windows_and_doors(opening_regions:list[Loop],
                                window_blocks:list[dict],
                                door_blocks:list[dict],
                                wall_lines:list[LineSeg],
                                watch_lines:list[LineSeg],
                                opening_lines:list[LineSeg],
                                rt_watch_opening_lines:STRTree,
                                opening_block_dist_limit:float,
                                )->tuple[list[Window],list[Door]]:
    """识别图中所有的窗和门"""
    windows:list[Window]=[]
    doors:list[Door]=[]
    # 门块
    for block in door_blocks:
        new_door=Door(parent=None,
                      type=block["block_name"],
                      width=block["scale"][0],
                      insert_point=Node(block["insert_point"][0],block["insert_point"][1]),
                      )
        doors.append(new_door)
    # 窗块
    for block in window_blocks:
        new_window=Window(parent=None,
                          type=block["block_name"],
                          width=block["scale"][0],
                          insert_point=Node(block["insert_point"][0],block["insert_point"][1]),
                          )
        windows.append(new_window)
    # 门窗线 & 门槛线
    lines_in_region={region:[] for region in opening_regions}
    for region in opening_regions:
        neighbor_lines:list[LineSeg]=rt_watch_opening_lines.query(region.get_aabb())
        for line in neighbor_lines:
            if line.length<const.TOL_DIST: continue
            mid=line.point_at(t=0.5)
            if not region.contains(mid): continue
            for region_bound in region.edges:
                if mid.is_on_LineSeg(region_bound): 
                    break
            else:
                lines_in_region[region].append(line)
                plt.plot([line.s.x,line.e.x],[line.s.y,line.e.y],color='k')
    # 将门窗块分配到距离最近的墙洞区域，但距离必须<=控制值
    # 如果最后有门窗块没有被分配，就原地创建一片墙，把门窗挂在上面
    for opening_block in doors+windows:
        min_dist=const._MAX_VAL
        nearest_region=None
        for region in opening_regions:
            centroid=LineSeg(region.nodes[0],region.nodes[2]).point_at(t=0.5)
            dist=centroid.dist(opening_block.insert_point)
            if dist<min_dist and dist<=opening_block_dist_limit:
                min_dist=dist
                nearest_region=region
        if nearest_region is not None:
            opening_block.region=nearest_region
        else:
            # new_region=Loop([opening_block.insert_point+]) TODO
            pass
    return windows,doors
                
def _draw_edge(line:LineSeg,base_color:str="m",edge_color:str="b",alpha:float=0.5):
    left=line.offset(line.lw)
    right=line.offset(-line.rw)
    def plt_line(line,c):plt.plot([line.s.x,line.e.x],[line.s.y,line.e.y],color=c,alpha=alpha)
    plt_line(line,base_color)
    plt_line(left,edge_color)
    plt_line(right,edge_color)

if __name__=="__main__":
    import json
    WALL_WIDTH_LIMITS=(100,300)  # 墙宽度范围：（下限,上限）
    RAIL_WIDTH_LIMITS=(30,60)  # 栏杆宽度范围：（下限，上限）
    BUTRESS_DIST_LIMITS=(10000,1500)  # 墙垛距离范围：(成对的墙垛距离上限，单个的墙垛距离上限)；下限=墙宽上限
    OPENING_BLOCK_DIST_LIMIT=1000
    const=Constant.default()

    with open("test/find_wall/case_2.json",'r',encoding="utf8") as f:
        j_data=json.load(f)

    wall_lines,watch_lines,opening_lines,window_blocks,door_blocks,block_defs=parse_data(j_data)

    # 合并看线（仅看线，不混合墙线，忽略交点） (OK)
    watch_lines=MergeEdgeAlgo(watch_lines,break_at_intersections=False).get_result()

    # 区分看线当中的窗和栏杆，当做门窗处理 (OK)
    watch_lines,new_opening_lines=remove_watch_window_and_railing(watch_lines,
                                                                  WALL_WIDTH_LIMITS,
                                                                  RAIL_WIDTH_LIMITS,
                                                                  )
    opening_lines.extend(new_opening_lines)

    # 合并墙线（仅墙线，不混合看线，在交点处打断） (OK)
    wall_lines=MergeEdgeAlgo(wall_lines,break_at_intersections=True).get_result()
    
    # 求墙线外轮廓 (OK)
    connected_graph=FindConnectedGraphAlgo(wall_lines).get_result()
    wall_outline_loops=[FindOutlineAlgo(g).get_result() for g in connected_graph]

    # 找墙垛 (OK)
    butresses=find_butress(wall_outline_loops)

    # 构建墙洞区域 (OK)
    opening_regions=find_opening_regions(butresses,
                                         wall_outline_loops,
                                         BUTRESS_DIST_LIMITS,
                                         )
    
    # 在交点处打断看线和门窗线，墙线不动 (OK)
    watch_lines,opening_lines,_=BreakEdgeAlgo(edge_groups=[watch_lines,opening_lines,wall_lines]).get_result()
    rt_watch_opening_lines=STRTree(watch_lines+opening_lines)

    # 识别门窗
    windows,doors=recognize_windows_and_doors(opening_regions,
                                              window_blocks,
                                              door_blocks,
                                              wall_lines,
                                              watch_lines,
                                              opening_lines,
                                              rt_watch_opening_lines,
                                              OPENING_BLOCK_DIST_LIMIT,
                                              )

    # 将门窗分配到墙洞区域
    # assign_windows_and_doors_to_opening_regions(windows,doors,opening_regions)

    # 在交点处打断线 (OK)
    watch_lines,wall_lines=BreakEdgeAlgo(edge_groups=[watch_lines,wall_lines]).get_result()

########################################### PLOT BEGIN ###########################################
    # from matplotlib.colors import TABLEAU_COLORS
    # colors=list(TABLEAU_COLORS)
    # for idx,g in enumerate(wall_outline_loops):
    #     color=colors[idx % len(colors)]
    #     for line in g.edges:
    #         plt.plot(*line.to_array().T,color=color)
    for line in wall_lines:
        _draw_edge(line,base_color="b",edge_color="b")
    for line in watch_lines:
        _draw_edge(line,base_color="r",edge_color="r")
    # for line in butresses:
    #     _draw_edge(line,base_color="m",edge_color="m",alpha=1)
    for loop in opening_regions:
        plt.fill(*loop.xy,color="g",alpha=0.2)
########################################### PLOT END ###########################################

    # 消除重叠的墙，优先保留wall_line  (X)不行，因为会有在窗台处画了墙线的情况，此时不能将窗台的看线消掉
    pass

    # # 墙线找墙
    # wall_line_groups=group_lines_by_angle(wall_lines)
    # walls=[]
    # for line_group in wall_line_groups:
    #     walls+=find_walls_from_parallel_lines(line_group,WALL_WIDTH_LIMIT)
    
    # # 处理门
        
    # # 看线找墙
    # watch_line_groups=group_lines_by_angle(watch_lines)
    # watch_walls=[]
    # for line_group in watch_line_groups:
    #     watch_walls+=find_walls_from_parallel_lines(line_group,WALL_WIDTH_LIMIT)

    # for wall in walls:
    #     _draw_edge(wall)
    # for wall in watch_walls:
    #     _draw_edge(wall)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
