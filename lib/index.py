import math
from lib.utils import Constant,ListTool
from lib.domain import Domain1d
from typing import Protocol,TYPE_CHECKING
if TYPE_CHECKING:
    from lib.geom import Geom,Node

class TreeNode[T]:
    """树结点"""
    def __init__(self,obj:T,parent:"TreeNode[T]"=None,child:list["TreeNode[T]"]=None) -> None:
        self.obj:T=obj
        self.parent:"TreeNode[T]"=parent
        self.child:list["TreeNode[T]"] =child[:] if child is not None else []

    def get_root(node:"TreeNode[T]")->"TreeNode[T]":
        while node.parent is not None:
            node=node.parent
        return node

class SupportCompare(Protocol):
    def __lt__(self,other)->bool:...
    def __le__(self,other)->bool:...
    def __gt__(self,other)->bool:...    
    def __ge__(self,other)->bool:...
    def __eq__(self,other)->bool:...
    def __ne__(self,other)->bool:...

class BSTNode[T:SupportCompare](TreeNode[T]):
    def __init__(self, obj: T, parent: "BSTNode[T]" = None, lch: "BSTNode[T]" = None, rch:"BSTNode[T]" = None) -> None:
        self.child:list["BSTNode[T]"]
        super().__init__(obj,parent,[lch,rch])
        self.count=1  # 结点上存储的数据数量
        self.size=1  # 树中结点的数量
        self.h=1  # 树高
    @property
    def lch(self)->"BSTNode[T]":
        return self.child[0]
    @lch.setter
    def lch(self,node:"BSTNode[T]")->None:
        self.child[0]=node
    @property
    def rch(self)->"BSTNode[T]":
        return self.child[1]    
    @rch.setter
    def rch(self,node:"BSTNode[T]")->None:
        self.child[1]=node

class BSTree[T:SupportCompare]:
    """二叉搜索树"""
    def __init__(self,objs:list[T]=None) -> None:
        if objs is None: objs=[]
        self.root:BSTNode[T]=None
        for obj in objs:
            self.insert(obj)
    def traverse(self)->list[T]:
        """顺序遍历"""
        if self.root is None: return []
        res=[]
        self._traverse(self.root,res)
        return res            
    def _traverse(self,root:BSTNode[T],res:list[T]):
        """顺序遍历根为root的子树，结果存入res"""
        if root.lch is not None: self._traverse(root.lch,res)
        for _ in range(root.count): res.append(root.obj)
        if root.rch is not None: self._traverse(root.rch,res)
    def insert(self,obj:T)->BSTNode[T]:
        """将值obj插入树中，返回插入的结点"""
        if self.root is None: 
            self.root=BSTNode(obj)
        else:
            self._insert(obj,self.root)
    def _insert(self,obj:T,root:BSTNode[T])->BSTNode[T]:
        """将值obj插入到根为root的子树中，返回插入的结点"""
        root.size+=1
        if obj==root.obj:
            root.count+=1
            return root
        i=0 if obj<root.obj else 1
        if root.child[i] is None:
            root.child[i]=BSTNode(obj,root)
            if root.h==1: root.h=2
            return root.child[i]
        else:
            new_node=self._insert(obj,root.child[i])
            root.h=1+max(root.lch.h if root.lch is not None else 0,
                         root.rch.h if root.rch is not None else 0)
            return new_node
    def find(self,obj:T)->BSTNode[T]|None:
        """查找值为obj的结点"""
        if self.root is None: return None
        return self._find(obj,self.root)
    def _find(self,obj:T,root:BSTNode[T])->BSTNode[T]|None:
        """在根为root的子树中查找值为obj的结点"""
        if obj==root.obj: 
            return root
        else: 
            ch=root.lch if obj<root.obj else root.rch
            return self._find(obj,ch) if ch is not None else None    
    def remove(self,obj:T)->None:
        """删除值为obj的结点"""
        if self.root is None: return
        self.root=self._remove(obj,self.root)
    def _remove(self,obj:T,root:BSTNode[T])->BSTNode[T]:
        """在根为root的子树中删除值为obj的结点，返回删除后的树根"""
        if obj==root.obj:
            if root.count>1:
                root.count-=1
            elif root.lch is None and root.rch is None:
                return None
            elif root.rch is None or root.lch is None:
                ch=root.lch if root.lch is not None else root.rch
                ch.parent=root.parent
                return ch
            else: 
                pred=self.get_max_node(root.lch)
                root.obj=pred.obj
                root.count=pred.count
                pred.count=1
                root.lch=self._remove(pred.obj,root.lch)
        else:
            i=0 if obj<root.obj else 1
            if root.child[i] is not None:
                root.child[i]=self._remove(obj,root.child[i])
        root.size=(root.count
                   +root.lch.size if root.lch is not None else 0
                   +root.rch.size if root.rch is not None else 0)
        root.h=1+max(root.lch.h if root.lch is not None else 0,
                     root.rch.h if root.rch is not None else 0)
        return root        
    def _l_rotate(self,root:BSTNode[T])->BSTNode[T]:
        """左旋，返回旋转后该位置的节点"""
        return self._rotate(root,0)
    def _r_rotate(self,root:BSTNode[T])->BSTNode[T]:
        """右旋，返回旋转后该位置的节点"""
        return self._rotate(root,1)
    def _rotate(self,root:BSTNode[T],i:int)->BSTNode[T]:
        """旋转，返回旋转后该位置的节点，i=0->左旋，i=1->右旋"""
        if root.child[1-i] is None: return root
        new_node=root.child[1-i]
        root.child[1-i]=new_node.child[i]
        root.child[1-i].parent=root
        new_node.parent=root.parent
        if root.parent is not None:
            if root is root.parent.child[i]: root.parent.child[i]=new_node
            else: root.parent.child[1-i]=new_node
        root.parent=new_node
        new_node.child[i]=root
        return new_node    
    def get_max_node(self,root:"BSTNode[T]")->"BSTNode[T]":
        """返回根为root的子树中最大的结点"""
        while root.rch is not None:
            root=root.rch
        return root
    def get_min_node(self,node:"BSTNode[T]")->"BSTNode[T]":
        """返回根为root的子树中最小的结点"""
        while node.lch is not None:
            node=node.lch
        return node

class UFSNode[T](TreeNode):
    """并查集"""
    def __init__(self, obj:T, parent: "UFSNode[T]" = None, child: list["UFSNode[T]"] = None) -> None:
        super().__init__(obj, parent, child)
    def get_root(self,node:"UFSNode[T]")->"UFSNode[T]":
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
        from lib.geom import Geom
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
        # 初始化叶子结点：几何图形的包围盒
        child_treenodes=[_STRNode(i,geom,None) for i,geom in enumerate(geoms)]  
        if len(geoms)==0: return
        # 每次循环自底向上构建一层树结构，直到最顶层一个根节点
        while True:
            child_num=len(child_treenodes)  # 本层子结点的数量
            parent_num=math.ceil(child_num/node_capacity)  # 本层父结点的数量
            slice_num=math.ceil(parent_num**0.5)  # 划分子结点的行列数
            col_cap=math.ceil(child_num/slice_num)  # 每列的子节点数
            # 子节点按x排序后划分成slice_num列
            child_treenodes.sort(key=lambda treenode: treenode.mbb[0].x)
            parent_treenodes=[]
            for i in range(slice_num):
                # 当前第i列的子节点们
                l,r=col_cap*i, col_cap*(i+1)
                if l>=child_num:break
                col=child_treenodes[l:r] if r<=child_num else child_treenodes[l:]
                # 每一列的子节点再按y排序后划分成slice_num个tile
                tile_cap=math.ceil(len(col)/slice_num)
                col.sort(key=lambda node: node.mbb[0].y)
                for j in range(slice_num):
                    b,t=tile_cap*j,tile_cap*(j+1)
                    if b>=len(col): break
                    tile=col[b:t] if t<=len(col) else col[b:]
                    # 为每个tile构建新的父节点
                    new_parent_node=_STRNode(index=-1,geom=None,child=tile)
                    for node in tile: node.parent=new_parent_node
                    parent_treenodes.append(new_parent_node)
            if len(parent_treenodes)==1:  # 到根节点停止
                self._root=parent_treenodes[0]
                break
            else:  # 否则继续构建上一层
                child_treenodes=parent_treenodes
    def __getitem__(self,index:int)->T:
        return self.geoms[index]
    def query_idx(self,extent:tuple["Node","Node"],tol:float=0,tree_node:_STRNode=None)-> list[int]:
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
    def query(self,extent:tuple["Node","Node"],tol:float=0,tree_node:_STRNode=None) -> list[T]:
        """框选查询

        Args:
            extent (tuple[Node,Node]): 范围矩形左下右上.
            tol (float, optional): 外扩距离. Defaults to 0.
            tree_node (_STRNode, optional): 开始查询的结点. Defaults to None->self._root.

        Returns:
            list[T]: 查询到的几何图形.
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
