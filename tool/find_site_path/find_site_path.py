if __name__=="__main__":
    import sys,os
    WORKING_DIR=os.getcwd()
    FILE_DIR=os.path.dirname(__file__)
    sys.path.append(WORKING_DIR)

import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from lib.geom import Node,Edge,Loop
from lib.index import TreeNode
from lib.utils import Timer
from tool.find_site_path.plot_helper  import plot_polygon

def read_excel(fname:str)->List[Node]:
    df=pd.read_excel(fname)
    nodes:List[Node] = []
    for index,row in df.iterrows():
        x1,y1=row["起点 X"],row["起点 Y"]
        x2,y2=row["端点 X"],row["端点 Y"]
        color='r' if row["颜色"]=="红" else 'k'
        if color=='k': continue
        lw,rw=0,0
        s=Node(x1,y1)
        e=Node(x2,y2)
        #去重
        if s.equals(e):continue
        sIsNew,eIsNew=True,True
        for node in nodes:
            if s.equals(node): 
                s=node
                sIsNew=False
            if e.equals(node): 
                e=node
                eIsNew=False
        if sIsNew: nodes.append(s)
        if eIsNew: nodes.append(e)
        #将边加入邻接表
        new_edge=Edge(s,e,lw,rw)
        new_edge.color=color
        s.add_edge_in_order(new_edge)
        new_edge=Edge(e,s,rw,lw)
        new_edge.color=color
        e.add_edge_in_order(new_edge)
    return nodes
def find_loop(nodes:List[Node])->List[Loop]:
    loops:List[Loop] =[]
    edge_num=0
    for node in nodes: #算总边数
        edge_num+=len(node.edge_out)
    while edge_num>0: #每次循环找一个环，直到所有边都被遍历过
        new_loop=[]
        for node in nodes: #先随便找一条边作为pre_edge
            if len(node.edge_out)>0:
                pre_edge=node.edge_out[0]
                break
        while True: #以pre_edge.e为当前结点开始找一个环
            node=pre_edge.e #当前结点
            theta=pre_edge.opposite().angle #入边的角度
            i=len(node.edge_out)-1
            while i>=0 and node.edge_out[i].angle>=theta: #按角度找下一条出边
                i-=1

            if node.edge_out[i].is_visited:  #如果找到了已访问的边就封闭这个环
                loops.append(Loop(new_loop)) #先将此环加入list
                for i in range(len(new_loop)): #并把环上的边都从邻接表里删掉
                    new_loop[i].s.edge_out.remove(new_loop[i])
                edge_num-=len(new_loop) #从总边数中减去环的边数
                break
            else: #如果找到的不是已访问的边
                new_loop.append(node.edge_out[i]) #就将此边加入环
                node.edge_out[i].is_visited=True #标记为已访问
                pre_edge=node.edge_out[i] #接着找下一条边
    return loops
# @Timer(__name__)
def make_cover_tree(loops:List[Loop])->List[TreeNode]:
    t:List[TreeNode] =[TreeNode(loop) for loop in loops] #把loop都变成TreeNode
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

if __name__=="__main__":
    nodes=read_excel(FILE_DIR+"\\test_data\\总图划分.xlsx")

    loops=find_loop(nodes)
    loop_tree=make_cover_tree(loops)

    for loop in loop_tree:
        color="r" if loop.area>0 else "y"
        for edge in loop.edges:
            plt.plot([edge.s.x,edge.e.x],[edge.s.y,edge.e.y],color=color)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
