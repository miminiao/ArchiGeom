import math
from lib.utils import Constant,ListTool
from lib.domain import Domain1d
from lib.geom import Geom,Node
# from lib.basic_structure import TreeNode

class TreeNode[T]:
    """树"""
    def __init__(self,obj:T,parent:"TreeNode[T]"=None,child:list["TreeNode[T]"]=None) -> None:
        self.obj:T=obj
        self.parent:"TreeNode[T]"=parent
        self.child:list["TreeNode[T]"] =child[:] if child is not None else []

    def getRoot(node:"TreeNode[T]")->"TreeNode[T]":
        while node.parent is not None:
            node=node.parent
        return node
    def get_leaf_nodes(self)->list["TreeNode[T]"]:
        nodes=[]
        for ch in self.child:
            if ch is not None:
                nodes.extend(self.get_leaf_nodes(ch))
        return nodes
    
class UFSNode[T](TreeNode):
    """并查集"""
    def __init__(self, obj:T, parent: "UFSNode[T]" = None, child: list["UFSNode[T]"] = None) -> None:
        super().__init__(obj, parent, child)
    def getRoot(self,node:"UFSNode[T]")->"UFSNode[T]":
        path=[]
        while node.parent is not None:
            path.append(node)
            node=node.parent
        for i in path: i.parent=node
        return node        
    def add(node:"UFSNode[T]",parent:"UFSNode[T]")->None:
        node.parent=parent
        parent.child.append(node)

class _STRNode[T:Geom](TreeNode):
    """STR树结点"""
    def __init__(self,index:int,geom:T=None,child:list["_STRNode[T]"]=None) -> None:
        super().__init__(geom,None,child)
        self.index=index
        self.mbb=geom.get_mbb() if index!=-1 else Geom.merge_mbb([ch.mbb for ch in child])

class STRTree[T:Geom]:
    """STR树"""
    def __init__(self, geoms:list[T], node_capacity:int=10) -> None:
        """以几何图形外包矩形构造一棵STR树

        Args:
            geoms (list[T:Geom]): 几何图形
            node_capacity (int, optional): 划分子结点的数量. Defaults to 10.
        """
        self.geoms=geoms
        child_treenodes=[_STRNode(i,geom,None) for i,geom in enumerate(geoms)] #初始化叶子结点：几何图形的mbb
        if len(geoms)==0: return
        while True: #每次循环自底向上构建一层树结构
            child_num=len(child_treenodes) #子结点的数量
            parent_num=math.ceil(child_num/node_capacity) #父结点的数量
            slice_num=math.ceil(parent_num**0.5) #划分儿子结点的行列数
            col_cap=math.ceil(child_num/slice_num) #每列的儿子节点数
            #儿子按x排序后划分成slice_num列
            child_treenodes.sort(key=lambda treenode: treenode.mbb[0].x)
            parent_treenodes=[]
            for i in range(slice_num):
                l,r=col_cap*i, col_cap*(i+1)
                if l>=child_num:break
                col=child_treenodes[l:r] if r<=child_num else child_treenodes[l:]
                #每一列的儿子按y排序后划分成slice_num个tile
                tile_cap=math.ceil(len(col)/slice_num)
                col.sort(key=lambda node: node.mbb[0].y)
                for j in range(slice_num):
                    b,t=tile_cap*j,tile_cap*(j+1)
                    if b>=len(col): break
                    tile=col[b:t] if t<=len(col) else col[b:]
                    new_node=_STRNode(index=-1,geom=None,child=tile)
                    for node in tile: node.parent=new_node   
                    parent_treenodes.append(new_node)
            if len(parent_treenodes)==1: #根节点
                self._root=parent_treenodes[0]
                break
            child_treenodes=parent_treenodes
    def __getitem__(self,index:int)->T:
        return self.geoms[index]
    def query_idx(self,extent:tuple[Node,Node],tol:float=0,tree_node:_STRNode=None)-> list[int]:
        """框选查询

        Args:
            extent (tuple[Node,Node]): 范围矩形左下右上.
            tol (float, optional): 外扩距离. Defaults to 0.
            tree_node (_STRNode, optional): 开始查询的结点. Defaults to None->self._root.

        Returns:
            list[int]: 查询到的几何图形index.
        """        
        tree_node=tree_node or self._root
        if tree_node.index!=-1:
            return [tree_node.index]
        res=[]
        qmin,qmax=extent
        for ch in tree_node.child:
            chmin,chmax=ch.mbb
            if (qmin.x<=chmax.x+tol) and (chmin.x<=qmax.x+tol) and (qmin.y<=chmax.y+tol) and (chmin.y<=qmax.y+tol):
                res=res+self.query_idx(extent,tol,ch)
        return res
    def query(self,extent:tuple[Node,Node],tol:float=0,tree_node:_STRNode=None) -> list[T]:
        """框选查询

        Args:
            extent (tuple[Node,Node]): 范围矩形左下右上.
            tol (float, optional): 外扩距离. Defaults to 0.
            tree_node (_STRNode, optional): 开始查询的结点. Defaults to None->self._root.

        Returns:
            list[T:Geom]: 查询到的几何图形.
        """
        return [self[i] for i in self.query_idx(extent,tol,tree_node)]

class SegmentTree:  # TODO
    def __init__(self,segs:list[Domain1d],const:Constant=None) -> None:
        """线段树(离线)

        Args:
            segs (list[Domain1d]): 待插入的线段
        """
        # 用所有区间的端点建立树结构，然后把区间逐个插到树里
        self.const=const or Constant.default()
        self.endpoints=[]
        for seg in segs: self.endpoints.extend([seg.l,seg.r])
        ListTool.sort_and_overkill(self.endpoints,self.const)
        self.root=self._construct_tree(0,len(self.endpoints)-1)
        for seg in segs: self._insert(self.root,seg)
    def _construct_tree(self,l:int,r:int)->TreeNode:
        """建立树结构

        Args:
            l (int): 当前根节点的左端点index
            r (int): 当前根节点的右端点index

        Returns:
            TreeNode: 当前根节点
        """
        # full_num=1<<int(math.ceil(math.log2(len(endpoints))))
        if l==r: return None
        node=TreeNode(Domain1d(self.endpoints[l],self.endpoints[r],-1))
        if r-l==1: return node
        node.child=[self._construct_tree(l,(l+r)//2),
                    self._construct_tree((l+r)//2,r),
                    ]
        for ch in node.child: ch.parent=node
        return node
    def _insert(self,node:TreeNode,seg:Domain1d)->None:
        """向树中插入一段新的线段

        Args:
            node (TreeNode): _description_
            seg (Domain1d): _description_
        """
        if node.parent is not None and node.obj.compare(node.parent.obj.value,node.obj.value)==1:  # 当前节点的父亲比儿子大，就刷新儿子
            node.obj.value=node.parent.obj.value
        if seg.l<node.obj.l and seg.r>node.obj.r:
            if node.obj.compare(seg.value,node.obj.value)==1:   # 新线段覆盖当前节点，且值比较大，就刷新当前节点
                node.obj.value=seg.value
            return
        if seg.l<node.child[0].obj.r:
            self._insert(node.child[0],seg)
        if seg.r>node.child[1].obj.l:
            self._insert(node.child[1],seg)
    def get_leaf_segs_value(self)->list[Domain1d]:
        ...

# %% 线段合并测试，带优先级比较
if 1 and __name__ == "__main__":
    import json,random
    import matplotlib.pyplot as plt
    from lib.geom import LineSeg,Edge
    const=Constant.default()

    # with open("./test/merge_line/case_1.json",'r',encoding="utf8") as f:
    #     j_obj=json.load(f)
    # edges:list[Edge]=[]
    # for ent in j_obj:
    #     if ent["object_name"]=="line" and ent["layer"]=="WALL":
    #         x1,y1,z1=ent["start_point"]
    #         x2,y2,z2=ent["end_point"]
    #         s=Node(x1,y1)
    #         e=Node(x2,y2)
    #         if s.equals(e):continue
    #         edges.append(Edge(s,e))

    doms:list[Domain1d]=[]
    limits=(0,10000,1000)
    random.seed(0)
    for i in range(10):
        l=random.random()*(limits[1]-limits[0])+limits[0]
        # r=random.random()*(limits[1]-limits[0])+limits[0]
        r=l.x+1000
        h=random.random()*limits[2]
        doms.append(Domain1d(l,r,h))

    plt.subplot(2,1,1)
    for i,dom in enumerate(doms):
        plt.plot([dom.l,dom.r],[dom.h,dom.h])

    print(f"{len(doms)} lines before")
    def compare(self,a:Domain1d,b:Domain1d): 
        if abs(a.value-b.value)<const.TOL_VAL: return 0
        elif a.value>b.value: return 1
        else: return -1
    segtree=SegmentTree(doms)

    
    # print(f"{len(merged_lines)} lines after")

    # plt.subplot(2,1,2)
    # for i,dom in enumerate(merged_lines):
    #     plt.plot([dom.s.x,dom.e.x],[dom.s.y+dom.lw+dom.rw,dom.e.y+dom.lw+dom.rw])

    # plt.show()

    # CASE_ID="6"

    # with open(f"./test/merge_line/case_{CASE_ID}.json",'w',encoding="utf8") as f:
    #     json.dump(lines,f,ensure_ascii=False,default=lambda x:x.__dict__)
    # with open(f"./test/merge_line/case_{CASE_ID}_out.json",'w',encoding="utf8") as f:
    #     json.dump(merged_lines,f,ensure_ascii=False,default=lambda x:x.__dict__)
