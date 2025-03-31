import math
from lib.utils import Constant,ListTool
from lib.interval import Interval1d,MultiInterval1d
from typing import Self,Callable,TYPE_CHECKING
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
class _BinaryTreeNode[T](TreeNode[T]):
    """二叉树结点"""
    def __init__(self, obj:T, parent:Self=None, lch:Self=None, rch:Self=None) -> None:
        super().__init__(obj, parent, [lch,rch])
    @property
    def lch(self)->"_BSTreeNode[T]":
        return self.child[0]
    @lch.setter
    def lch(self,node:"_BSTreeNode[T]")->None:
        self.child[0]=node
    @property
    def rch(self)->"_BSTreeNode[T]":
        return self.child[1]    
    @rch.setter
    def rch(self,node:"_BSTreeNode[T]")->None:
        self.child[1]=node
class _BSTreeNode[T](_BinaryTreeNode[T]):
    """二叉搜索树结点"""
    def __init__(self, obj: T, parent: "_BSTreeNode[T]" = None, lch: "_BSTreeNode[T]" = None, rch:"_BSTreeNode[T]" = None) -> None:
        self.child:list["_BSTreeNode[T]"]
        super().__init__(obj,parent,[lch,rch])
        self.count=1  # 结点上存储的数据数量
        self._h=1  # 树高
    @classmethod
    def get_h(cls,node:"_BSTreeNode[T]")->int:
        """计算根为root的树高，需确保子树都已经计算好树高"""
        return 0 if node is None else node._h
    def update_h(self)->None:
        """更新树高，需确保子树都已经计算好树高"""
        self._h=1+max(_BSTreeNode.get_h(self.lch),_BSTreeNode.get_h(self.rch))
    def get_pred(self)->"_BSTreeNode[T]":
        """前驱结点"""
        pred=self.lch
        while pred.rch is not None:
            pred=pred.rch
        return pred
    def get_succ(self)->"_BSTreeNode[T]":
        """后继结点"""
        succ=self.rch
        while succ.lch is not None:
            succ=succ.lch
        return succ
    def _l_rotate(self)->"_BSTreeNode[T]":
        """左旋，返回旋转后原位置上的节点"""
        return self._rotate(0)
    def _r_rotate(self)->"_BSTreeNode[T]":
        """右旋，返回旋转后原位置上的节点"""
        return self._rotate(1)
    def _rotate(self,i:int)->"_BSTreeNode[T]":
        """旋转，返回旋转后原位置上的节点，i=0->左旋，i=1->右旋"""
        if self.child[1-i] is None: return self
        new_node=self.child[1-i]
        self.child[1-i]=new_node.child[i]
        if self.child[1-i] is not None: self.child[1-i].parent=self
        new_node.parent=self.parent
        if self.parent is not None:
            if self is self.parent.child[i]: self.parent.child[i]=new_node
            else: self.parent.child[1-i]=new_node
        self.parent=new_node
        new_node.child[i]=self
        self.update_h()
        new_node.update_h()
        return new_node

class BSTree[T]:
    """二叉搜索树"""
    def __init__(self,objs:list[T]=None) -> None:
        if objs is None: objs=[]
        self.root:_BSTreeNode[T]=None
        for obj in objs:
            self.insert(obj)
    def traverse(self)->list[T]:
        """顺序遍历值"""
        return [node.obj for node in self.traverse_nodes()]
    def traverse_nodes(self)->list[_BSTreeNode[T]]:
        """顺序遍历节点"""
        res=[]
        self._traverse(self.root,res)
        return res
    @classmethod
    def _traverse(cls,root:_BSTreeNode[T],res:list[_BSTreeNode[T]])->None:
        """顺序遍历根为root的子树，结果存入res"""
        if root is None: return
        cls._traverse(root.lch,res)
        for _ in range(root.count): res.append(root)
        cls._traverse(root.rch,res)
    def insert(self,obj:T)->None:
        """将值obj插入树中"""
        self.root=self._insert(obj,self.root)
    @classmethod
    def _insert(cls,obj:T,root:_BSTreeNode[T],parent:_BSTreeNode[T]=None)->_BSTreeNode[T]:
        """将值obj插入到根为root的子树中，返回根结点"""
        if root is None: return _BSTreeNode(obj,parent=parent)
        if obj<root.obj:
            root.lch=cls._insert(obj,root.lch,parent=root)
        elif obj>root.obj:
            root.rch=cls._insert(obj,root.rch,parent=root)
        else:
            root.count+=1
        root.update_h()
        return root
    def find(self,obj:T)->_BSTreeNode[T]|None:
        """查找值为obj的结点"""
        if self.root is None: return None
        return self._find(obj,self.root)
    @classmethod
    def _find(cls,obj:T,root:_BSTreeNode[T])->_BSTreeNode[T]|None:
        """在根为root的子树中查找值为obj的结点"""
        if obj==root.obj: 
            return root
        else: 
            ch=root.lch if obj<root.obj else root.rch
            return cls._find(obj,ch) if ch is not None else None    
    def remove(self,obj:T)->None:
        """删除值为obj的结点"""
        if self.root is None: return
        self.root=self._remove(obj,self.root)
    @classmethod
    def _remove(cls,obj:T,root:_BSTreeNode[T])->_BSTreeNode[T]:
        """在根为root的子树中删除值为obj的结点，返回删除后的树根"""
        if root is None: return None
        if obj<root.obj:
            root.lch=cls._remove(obj,root.lch)
        elif obj>root.obj:
            root.rch=cls._remove(obj,root.rch)
        else:
            if root.count>1:
                root.count-=1
            elif root.lch is None and root.rch is None:
                return None
            elif root.lch is None:
                root.rch.parent=root.parent
                root=root.rch
            elif root.rch is None:
                root.lch.parent=root.parent
                root=root.lch
            else: 
                pred=root.get_pred()
                root.obj=pred.obj
                root.count=pred.count
                pred.count=1
                root.lch=cls._remove(pred.obj,root.lch)
        root.update_h()
        return root
    
class AVLTree[T](BSTree[T]):
    """AVL树"""
    def __init__(self, objs:list[T]=None) -> None:
        super().__init__(objs)
    @classmethod
    def _maintain_balance(cls,root:_BSTreeNode[T])->_BSTreeNode[T]:
        """维护root结点处的平衡"""
        if root is None: return None
        if _BSTreeNode.get_h(root.lch)-_BSTreeNode.get_h(root.rch)>1:
            if _BSTreeNode.get_h(root.lch.lch)>=_BSTreeNode.get_h(root.lch.rch):
                root=root._r_rotate()
            else:
                root.lch=root.lch._l_rotate()
                root=root._r_rotate()
        elif _BSTreeNode.get_h(root.rch)-_BSTreeNode.get_h(root.lch)>1:
            if _BSTreeNode.get_h(root.rch.rch)>=_BSTreeNode.get_h(root.rch.lch):
                root=root._l_rotate()
            else:
                root.rch=root.rch._r_rotate()
                root=root._l_rotate()
        return root
    @classmethod
    def _insert(cls,obj:T,root:_BSTreeNode[T],parent:_BSTreeNode[T]=None)->_BSTreeNode[T]:
        """将值obj插入到根为root的子树中，返回根结点"""
        root=super()._insert(obj,root,parent=parent)
        root=cls._maintain_balance(root)
        return root
    @classmethod
    def _remove(cls,obj:T,root:_BSTreeNode[T])->_BSTreeNode[T]:
        """在根为root的子树中删除值为obj的结点，返回删除后的树根"""
        root=super()._remove(obj,root)
        root=cls._maintain_balance(root)
        return root      
    
class _DSUNode[T](TreeNode[T]):
    def __init__(self, obj:T) -> None:
        super().__init__(obj)
    def get_root(self)->"_DSUNode[T]":
        if self.parent is not None:
            self.parent=self.get_root(self.parent)
        return self.parent
    
class DSU[T]:
    """并查集"""
    def __init__(self, objs:list[T]=None) -> None:
        self._nodes={obj:_DSUNode(obj) for obj in objs}
    def unite(self,s1:T,s2:T):
        self.find(s1).parent=self.find(s2)
    def find(self,obj:T)->"_DSUNode[T]":
        return self._nodes[obj].get_root()

class _STRTreeNode[T:Geom](TreeNode):
    """STR树结点"""
    def __init__(self,geom:T=None,child:list["_STRTreeNode[T]"]=None) -> None:
        from lib.geom import Geom
        super().__init__(geom,None,child)
        self.mbb=geom.get_mbb() if geom is not None else Geom.merge_mbb([ch.mbb for ch in child])

class STRTree[T:Geom]:
    """STR树"""
    def __init__(self, geoms:list[T], node_capacity:int=10) -> None:
        """以几何图形外包矩形构造一棵STR树.

        Args:
            geoms (list[T:Geom]): 几何图形
            node_capacity (int, optional): 划分子结点的数量. Defaults to 10.
        """
        self.geoms=geoms
        # 初始化叶子结点：几何图形的包围盒
        child_treenodes=[_STRTreeNode(geom,None) for geom in geoms]  
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
                    new_parent_node=_STRTreeNode(geom=None,child=tile)
                    for node in tile: node.parent=new_parent_node
                    parent_treenodes.append(new_parent_node)
            if len(parent_treenodes)==1:  # 到根节点停止
                self._root=parent_treenodes[0]
                break
            else:  # 否则继续构建上一层
                child_treenodes=parent_treenodes
    def query(self,extent:tuple["Node","Node"],tol:float=0,tree_node:_STRTreeNode=None) -> list[T]:
        """框选查询.

        Args:
            extent (tuple[Node,Node]): 范围矩形左下右上.
            tol (float, optional): 外扩距离. Defaults to 0.
            tree_node (_STRNode, optional): 开始查询的结点. Defaults to None->self._root.

        Returns:
            list[T]: 查询到的几何图形.
        """
        tree_node=tree_node or self._root
        if tree_node.obj is not None:
            return [tree_node.obj]
        res=[]
        qmin,qmax=extent
        for ch in tree_node.child:
            chmin,chmax=ch.mbb
            if (qmin.x<=chmax.x+tol) and (chmin.x<=qmax.x+tol) and (qmin.y<=chmax.y+tol) and (chmin.y<=qmax.y+tol):
                res=res+self.query(extent,tol,ch)
        return res        

class SegmentTree[T]:
    """线段树.

    Args:
        segs (list[Interval1d[T]]): 用于构造线段树的线段。端点的比较使用Interval1d._cmp.
    """
    def __init__(self,segs:list[Interval1d[T]]) -> None:
        # 用所有区间的端点建立树结构，然后把区间逐个插到树里
        self._cmp=Interval1d._cmp
        endpoints=[]
        for seg in segs: endpoints.extend([seg.l,seg.r])
        self.endpoints=ListTool.distinct(endpoints,cmp_func=self._cmp)
        # self.endpoints.sort()
        self.root=self._construct_tree(0,len(self.endpoints)-1)
        for seg in segs: self.insert(self.root,seg)
    def _construct_tree(self,l:int,r:int)->_BinaryTreeNode[Interval1d]:
        """在segs[l..r]上建立树结构，返回根节点"""
        if l==r: return None
        node=_BinaryTreeNode(Interval1d(self.endpoints[l],self.endpoints[r],None))
        if r-l==1: return node
        node.child=[self._construct_tree(l,(l+r)//2),
                    self._construct_tree((l+r)//2,r),
                    ]
        for ch in node.child: ch.parent=node
        return node
    @classmethod
    def _update_value(self,node:_BinaryTreeNode[Interval1d])->None:
        if node.parent is None or node.parent.obj.value is None: return
        if node.obj.value is None or node.parent.obj.value>node.obj.value:
            node.obj.value=node.parent.obj.value
    def insert(self,root:_BinaryTreeNode[Interval1d],seg:Interval1d)->None:
        """向根为root的子树中插入一段新的线段.

        Args:
            root (_BinaryTreeNode[Interval1d]): 当前树根.
            seg (Interval1d): 新线段.
        """
        self._update_value(root)  # lazy-update当前结点的value
        if root.obj.value is not None and seg.value<=root.obj.value:  # 新线段没现在的大，就不用看了
            return
        if self._cmp(seg.l,root.obj.l)<=0 and self._cmp(seg.r,root.obj.r)>=0:  # 新线段覆盖当前节点，就刷新当前节点
            root.obj.value=seg.value
            return
        if self._cmp(seg.l,root.lch.obj.r)<0:
            self.insert(root.lch,seg)
        if self._cmp(seg.r,root.rch.obj.l)>0:
            self.insert(root.rch,seg)
    @classmethod
    def _traverse_leaves(cls,root:_BinaryTreeNode[Interval1d],res:list[Interval1d])->None:
        cls._update_value(root)  # lazy-update当前结点的value
        if len(root.child)==0:
            res.append(root.obj)
        else:
            cls._traverse_leaves(root.child[0],res)
            cls._traverse_leaves(root.child[1],res)
    def get_united_leaves(self)->list[Interval1d]:
        """获取叶子结点区间合并的结果"""
        leaves:list[Interval1d]=[]
        self._traverse_leaves(self.root,leaves)
        res=[leaves[0]]
        for node in leaves:
            if node.value==res[-1].value:
                res[-1].r=node.r
            else:
                res.append(node)
        res=[obj for obj in res if obj.value is not None]
        return res
class _KDTreeNode[T](_BinaryTreeNode):
    def __init__(self, obj, dim:int, parent:Self=None, lch:Self=None, rch:Self=None):
        super().__init__(obj, parent, lch, rch)
        self.dim=dim  # 当前结点的切割方向
class KDTree[T]:
    """k-d树"""
    def __init__(self, nodes:list[Node], dim:int=2):
        self.dim=dim
        self._key=[lambda node:node[i] for i in range(dim)]
        self.root=self._construct_tree(nodes[:],0)
    def _construct_tree(self,nodes:list[Node],current_dim:int)->_KDTreeNode:
        """在nodes上建立kdtree，返回切割方向为current_dim的根结点"""
        if len(nodes)==1: return _KDTreeNode(nodes[0],dim=current_dim)
        m=len(nodes)//2
        median_idx=ListTool.get_nth(nodes,m,self._key[current_dim])  # 中位数的index
        nodes[median_idx],nodes[len(nodes)//2]=nodes[len(nodes)//2],nodes[median_idx]
        return _KDTreeNode()