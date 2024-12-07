"""平面最短路径"""

import pandas as pd
import matplotlib.pyplot as plt
from lib.geom import Node,Edge,Loop,Polygon,_draw_polygon
from lib.index import TreeNode
from lib.utils import Timer, Constant
    
def find_or_insert_node(target_node:Node,nodes:list[Node])->Node:
    for node in nodes:
        if target_node.equals(node):
            return node
    nodes.append(target_node)
    return target_node

def find_loop(nodes:list[Node])->list[Loop]:
    loops:list[Loop] =[]
    visited_edges=set()
    edge_num=sum([len(node.edge_out) for node in nodes]) #算总边数
    while edge_num>0: #每次循环找一个环，直到所有边都被遍历过
        new_loop:list[Node]=[]
        for node in nodes: #先随便找一条边作为pre_edge
            if len(node.edge_out)>0:
                pre_edge=node.edge_out[0]
                break
        while True: #以pre_edge.e为当前结点开始找一个环
            node=pre_edge.e #当前结点
            theta=pre_edge.opposite().angle #入边的角度
            i=len(node.edge_out)-1
            while i>=0 and node.edge_out[i].angle+const.TOL_ANG>=theta: #按角度找下一条出边
                i-=1
            if node.edge_out[i] in visited_edges:  #如果找到了已访问的边就封闭这个环
                loops.append(Loop(new_loop)) #先将此环加入list
                for i in range(len(new_loop)): #并把环上的边都从邻接表里删掉
                    new_loop[i].s.edge_out.remove(new_loop[i])
                edge_num-=len(new_loop) #从总边数中减去环的边数
                break
            else: #如果找到的不是已访问的边
                new_loop.append(node.edge_out[i]) #就将此边加入环
                visited_edges.add(node.edge_out[i]) #标记为已访问
                pre_edge=node.edge_out[i] #接着找下一条边
    return loops
def make_cover_tree(loops:list[Loop])->list[TreeNode]:
    t:list[TreeNode] =[TreeNode(loop) for loop in loops] #把loop都变成TreeNode
    for i in range(len(t)-1):
        for j in range(i+1,len(t)):
            ni,nj=t[i],t[j]
            ci=ni.obj.contains(nj.obj)
            cj=nj.obj.contains(ni.obj)
            if not ci and not cj: continue #没有覆盖关系时，跳过
            if cj and not ci: #j覆盖i且i不覆盖j（ij不重合）时，ij互换 
                ni,nj=nj,ni
            if (nj.parent is None) or (abs(ni.obj.area)<abs(nj.parent.obj.area)): #此时可确保i覆盖j，通过比较面积更新j.parent
                nj.parent=ni
    for i in t:
        if i.parent is not None:
            i.parent.child.append(i)
    return t

def get_door_points(nodes:list[Node],door_lines:list[Edge],point_num:int) -> dict[Edge:list[Node]]:
    """返回dict{门线：[门上各点]}"""
    door_points={}
    for door_line in door_lines:
        for node in nodes:
            for edge in node.edge_out:
                # 如果门线与墙相交，就记录门线对应的点
                if door_line.intersects(edge): 
                    mid=door_line.intersection(edge)
                    if point_num<=1:
                        door_points[door_line]=[mid]
                    else:
                        wall_norm=edge.to_vec3d().unit()
                        s=Node.from_vec3d(mid.to_vec3d()-wall_norm*door_line.lw)
                        e=Node.from_vec3d(mid.to_vec3d()+wall_norm*door_line.rw)
                        door_edge=Edge(s,e)  # 真实的开门范围
                        door_points[door_line]=[s]
                        for i in range(1,point_num):
                            door_point=door_edge.point_at(t=i/(point_num-1))[0]
                            door_points[door_line].append(door_point)
                    break
            else: continue
            break
    return door_points
def find_doors(loops:list[Loop],door_lines:list[Edge],door_points:dict[Edge:list[Node]],point_num:int) -> tuple[dict[Loop:list[Node]],list[list[Node]]]:
    """把门上的点分配到所在的loop上，然后把成对的点连起来"""
    door_point_pairs=[]  # 用相邻房间轮廓上的一对连通的结点表示门
    doors_on_loop={loop:[] for loop in loops}  # 保存每个房间里的门集合
    for door_line in door_lines:
        new_pairs=[[] for _ in range(point_num)]  # 门上各点对应的结点对
        for loop in loops:
            for edge in loop.edges:
                # 如果房间轮廓与门线相交，就取edge离门上各点最近的点
                if door_line.intersects(edge): 
                    for i in range(point_num):
                        new_node=edge.offset(0.1).closest_point(door_points[door_line][i])
                        new_pairs[i].append(new_node)
                        doors_on_loop[loop].append(new_node)
            if len(new_pairs[0])==2:
                door_point_pairs+=new_pairs
                break
    return doors_on_loop, door_point_pairs
def get_visibility_graph(room:Polygon,doors_on_loop:dict[Loop:list[Node]]) ->dict[Node:list[Node]]:
    """计算房间内可见性图"""
    nodes=room.nodes()
    loops=[room.exterior]+room.interiors
    # 加入门上的点
    for loop in loops:
        if loop in doors_on_loop:
            nodes+=doors_on_loop[loop]
    vis_graph={node:set() for node in nodes}  # 可见性图（邻接表）
    # 加入可直达的边
    for i in range(len(nodes)):
        for j in range(i):
            vis_line=Edge(nodes[i],nodes[j])
            if room.covers(vis_line):
                vis_graph[nodes[i]].add(nodes[j])
                vis_graph[nodes[j]].add(nodes[i])
    return vis_graph
def shortest_path(vis_graph:dict[Node:set[Node]],source:Node) -> tuple[dict[Node:float],dict[Node:Node]]:
    """dijkstra
        d: 各节点最短路径长度
        pre: 各节点最短路径上的前驱节点
    """
    d={node:const.MAX_VAL for node in vis_graph}
    d[source]=0
    pre={}
    safe_set=set()
    for i in range(len(vis_graph)-1):
        min_d=const.MAX_VAL
        for node in vis_graph:
            if node not in safe_set and d[node]<min_d:
                min_d=d[node]
                safe_node=node
        if min_d==const.MAX_VAL:
            break
        safe_set.add(safe_node)
        for node in vis_graph[safe_node]:
            if d[node]>d[safe_node]+safe_node.dist(node):
                d[node]=d[safe_node]+safe_node.dist(node)
                pre[node]=safe_node
    return d,pre

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
            for seg in obj["segments"]:
                x1,y1,_=seg["start_point"]
                x2,y2,_=seg["end_point"]
                lw=rw=seg["start_width"]/2
                bulge=seg["bulge"]
                s=Node(x1,y1)
                e=Node(x2,y2)
                if s.equals(e):continue
                s=find_or_insert_node(s,nodes)
                e=find_or_insert_node(e,nodes)
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

############################ 画线 ############################

    with Timer(tag="画图"):        
        # 画墙基线
        for loop in loops:
            for edge in loop.edges:
                other,t=edge.point_at(0.3)
                plt.plot([edge.s.x,other.x],[edge.s.y,other.y],color="m")
        # 画房间
        for room in rooms:
            _draw_polygon(room.polygon,color=('y','g'))
        # 画门
        pass
        # 画视线
        for s in vis_graph:
            for e in vis_graph[s]:
                plt.plot([s.x,e.x],[s.y,e.y],color='k',alpha=0.1)
        # 画路径
        if d[ST_PAIR[1]]<const.MAX_NUM:
            path=[ST_PAIR[1]]
            while path[-1] is not ST_PAIR[0]:
                path.append(pre[path[-1]])
            plt.plot([node.x for node in path],[node.y for node in path],color='r')
            
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
