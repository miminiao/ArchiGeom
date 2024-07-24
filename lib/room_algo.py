import pandas as pd
import matplotlib.pyplot as plt
from lib.geom import Node,Edge,Loop,Poly,_draw_polygon
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
    rooms:list[Poly]=[]
    for tree_node in cover_tree:
        if tree_node.obj.area>0:
            new_shell=tree_node.obj
            new_holes=[ch.obj for ch in tree_node.child]
            rooms.append(Poly(new_shell,new_holes))

    with Timer(tag="画图"):
        # 画墙基线
        for loop in loops:
            for edge in loop.edges:
                other,t=edge.point_at(0.3)
                plt.plot([edge.s.x,other.x],[edge.s.y,other.y],color="m")
        # 画房间
        for room in rooms:
            _draw_polygon(room.polygon,color=('y','g'))
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
