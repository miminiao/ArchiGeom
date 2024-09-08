求并:bool=True
求交:bool=not 求并

class Loop:
    def __init__(self,id:int,is_outer:bool,children:list["Loop"]) -> None:
        self.id=id
        self.is_outer=is_outer
        self.children=children
class Polygon:
    def __init__(self,outer:Loop,inners:list[Loop]) -> None:
        self.outer=outer
        self.inners=inners

n:int # 原始polygon的数量
roots:list[Loop]
polygons=list[Polygon]
def find_polygon(root:Loop, depth:int)->list[Loop]: # (根节点, 深度值) -> root的后代中还没人配对的节点们
    if root.is_outer:
        inners:list[Loop] # root的后代中与root配对的节点们
        for ch in root.children:
            inners+=find_polygon(ch,depth+1)
        if (求并 and depth==0)or(求交 and depth==n): 
            polygons.append(Polygon(root,inners))
        res:list[Loop]=[]
        for inner in inners:
            res+=find_polygon(inner,depth-1)
            return res
    else: return root

for root in roots:
    find_polygon(root,0)

loops=[Loop(i,True,[]) for i in range(22)]

roots=[Loop(1,True)]