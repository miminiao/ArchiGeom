class TreeNode[T]:
    """树"""
    def __init__(self,obj:T,parent:"TreeNode"[T]=None,child:list["TreeNode"[T]]=None) -> None:
        self.obj:T=obj
        self.parent:"TreeNode"[T]=parent
        self.child:list["TreeNode"[T]] =child[:] if child is not None else []

    def getRoot(node:"TreeNode"[T])->"TreeNode"[T]:
        while node.parent is not None:
            node=node.parent
        return node
    def get_leaf_nodes(self)->list["TreeNode"[T]]:
        nodes=[]
        for ch in self.child:
            if ch is not None:
                nodes.extend(self.get_leaf_nodes(ch))
        return nodes
class UFSNode[T](TreeNode):
    """并查集"""
    def __init__(self, obj:T, parent: "UFSNode"[T] = None, child: list["UFSNode"[T]] = None) -> None:
        super().__init__(obj, parent, child)
    def getRoot(self,node:"UFSNode"[T])->"UFSNode"[T]:
        path=[]
        while node.parent is not None:
            path.append(node)
            node=node.parent
        for i in path: i.parent=node
        return node        
    def add(node:"UFSNode"[T],parent:"UFSNode"[T])->None:
        node.parent=parent
        parent.child.append(node)