from abc import ABC,abstractmethod
from lib.geom import Node,Edge,Loop
from lib.linalg import Vec3d,Mat3d

class BuildingElement(ABC):
    def __init__(self) -> None:
        """建筑构件"""
        pass

class Wall(BuildingElement):
    def __init__(self,base:Edge,lw:float=100,rw:float=100,h:float=3000.0) -> None:
        """墙

        Args:
            base_edge (Edge): 墙基线.
            lw (float, optional): 左宽. Defaults to 100.
            rw (float, optional): 右宽. Defaults to 100.
            h (float, optional): 墙高. Defaults to 3000.
        """
        super().__init__()
        self.base=base
        self.lw,self.rw,self.h=lw,rw,h
        self.windows:list[Window]=[]
        self.doors:list[Door]=[]
        self.holes:list[Hole]=[]
    def insert_opening(self,opening:"Opening")->None:
        opening.parent=self
        translation=opening.insert_point.to_vec3d()-self.base.s.to_vec3d()
        basis=self.base.frame_at(self.base.get_point_param(opening.insert_point))
        opening.pos_wrt_wall=basis.transpose()@translation
        type_map={Window:self.windows,Door:self.doors,Hole:self.holes}
        type_map[type(opening)].append(opening)
    def insert_opening_wrt_wall_basis(self,opening:"Opening",pos_wrt_wall_basis:Vec3d)->None:
        opening.parent=self
        opening.pos_wrt_wall=pos_wrt_wall_basis
        opening.insert_point=self.s.to_vec3d()+self.basis@pos_wrt_wall_basis
        type_map={Window:self.windows,Door:self.doors,Hole:self.holes}
        type_map[type(opening)].append(opening)
class Opening(BuildingElement):
    def __init__(self,parent:Wall,type:str,width:float,insert_point:Node,region:Loop,axis_reverse=None) -> None:
        super().__init__()
        self.parent:Wall=parent
        self.type:str=type
        self._insert_point:Node=insert_point
        self.width:float=width
        self.region:Loop=region
        self.pos_wrt_wall:Vec3d=None
        self.axis_reverse=axis_reverse or [False,False,False]
    @property
    def insert_point(self)->Node:
        # if self.pos_wrt_wall is None: 
            return self._insert_point
    def start_point(self)->Node:
        self.parent.base.point_at(self.parent.base.get_point_param(self.insert_point))
        return 
class Door(Opening):
    def __init__(self, parent: Wall, type: str, width: float, insert_point: Node=None, region: Loop=None) -> None:
        super().__init__(parent, type, width, insert_point, region)
    
class Window(Opening):
    def __init__(self, parent: Wall, type: str, width: float, insert_point: Node=None, region: Loop=None) -> None:
        super().__init__(parent, type, width, insert_point, region)

class Hole(Opening):
    def __init__(self, parent: Wall, type: str, width: float, insert_point: Node=None, region: Loop=None) -> None:
        super().__init__(parent, type, width, insert_point, region)

class RoomProfileEdge(BuildingElement):
    def __init__(self,edge:Edge,wall:Wall) -> None:
        super().__init__()
        self.edge=edge
        self.wall=wall
    def relative_orientation(self)->int:
        """边与墙的相对方向

        Returns:
            int: 1:同向平行; -1:反向平行; 0:不平行.
        """
        e1,e2=self.edge,self.wall.base
        if not e1.is_collinear(e2): return 0
        if e1.is_zero() or e2.is_zero(): return 1
        return e1.tangent_at(0.5).dot(e2.tangent_at(0.5))>0
class RoomProfile(BuildingElement):
    def __init__(self,walls:list[Wall]) -> None:
        """描述房间轮廓，

        Args:
            walls (list[Wall]): _description_
        """
        Loop.__init__(self,walls)

class Room(BuildingElement):
    def __init__(self,inner_profile:RoomProfile) -> None:
        BuildingElement.__init__(self)
        self.inner_profile=inner_profile
    @property
    def walls(self)->list[Wall]:
        return [wall for wall in self.inner_profile.edges]

if __name__=="__main__":
    wall=Wall(Edge(Node(200,300),Node(1200,800)))
    window=Window(wall,"1231456",500,Node(700,800),)
    wall.insert_opening(window)
    print(window.pos_wrt_wall)
    wall.insert_opening_wrt_wall_basis(window,Vec3d(500,100,0))
    print(window.insert_point)
