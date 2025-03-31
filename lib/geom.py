#%%
import numpy as np
import math
from copy import copy,deepcopy
from enum import Enum
from abc import ABC,abstractmethod
from typing import Self,Generator,overload
from lib.utils import Timer,Constant as Const
from lib.linalg import Vec3d,Mat3d
from lib.index import STRTree

class GeomRelation(Enum):
    """几何对象other和self的关系"""
    Inside=1  # other与self的内部有交集
    Outside=2  # other与self的外部有交集
    OnBoundary=3  # other与self的边界有重叠
    Intersect=4  # other与self的边界有交点

class Geom(ABC):
    _dumper_ignore=[]
    def __init__(self) -> None: ...
    @staticmethod
    def merge_mbb(mbbs:list[tuple["Node","Node"]]) -> tuple["Node","Node"]:
        """合并包围盒，返回一堆包围盒的包围盒.

        Args:
            mbbs (list[tuple[Node,Node]]): 一堆包围盒.
        Returns:
            tuple[Node,Node]: 大包围盒.
        """
        if len(mbbs)==0: return None
        max_val=Const.MAX_VAL
        pmin=Node(max_val,max_val)
        pmax=Node(-max_val,-max_val)
        for mbb in mbbs:
            pmin.x=min(pmin.x,mbb[0].x)
            pmin.y=min(pmin.y,mbb[0].y)
            pmax.x=max(pmax.x,mbb[1].x)
            pmax.y=max(pmax.y,mbb[1].y)
        return (pmin,pmax)
    
    @staticmethod
    def mbb_relation(a:tuple["Node","Node"],b:tuple["Node","Node"])->list[GeomRelation]:
        rel=[]
        comp=Const.cmp_dist
        if (comp(b[0].x,a[1].x)<0 and comp(b[0].y,a[1].y)<0 and
            comp(b[1].x,a[0].x)>0 and comp(b[1].y,a[0].y)>0
        ): 
            rel.append(GeomRelation.Inside)
        if (comp(b[0].x,a[0].x)<0 or comp(b[0].y,a[0].y)<0 or
            comp(b[1].x,a[1].x)>0 or comp(b[1].y,a[1].y)>0
        ): 
            rel.append(GeomRelation.Outside)
        return rel
    @abstractmethod
    def get_mbb(self)->tuple["Node","Node"]:
        """获取包围盒->(左下,右上)"""
        ...
    
class Node(Geom):
    """点"""
    _dumper_ignore=["edge_out","edge_in"]
    def __init__(self, x:float, y:float, z:float=None) -> None:
        super().__init__()
        self.x=x
        self.y=y
        self.z=z or 0
        self.edge_out:list[Edge]=[]
        self.edge_in:list[Edge]=[]
    def __getitem__(self,i:int)->float:
        return (self.x,self.y,self.z)[i]
    def __repr__(self) -> str:
        return f"Node({round(self.x,2)},{round(self.y,2)})"
    def __eq__(self,other):
        return self.equals(other)
    def __hash__(self)->int:
        return id(self)
    def __add__(self,other:Vec3d)->Self:
        return Node.from_vec3d(self.to_vec3d()+other)
    def __sub__(self,other:Vec3d)->Self:
        return Node.from_vec3d(self.to_vec3d()-other)
    def __copy__(self)->Self: 
        return Node(self.x,self.y,self.z)
    def __deepcopy__(self):
        return Node(self.x,self.y,self.z)
    @classmethod
    def from_array(cls,arr:np.ndarray) -> Self:
        if arr.shape==(2,):
            return cls(arr[0],arr[1])
        else: return None
    @classmethod
    def from_vec3d(cls,vec:Vec3d) -> Self:
        return cls(vec.x,vec.y)
    def get_mbb(self) -> tuple[Self, Self]:
        return (self,self)
    def equals(self, other:Self) -> bool:
        return isinstance(other,Node) and self.dist(other)<Const.TOL_DIST
    def dist(self, other:Self) -> bool:
        return ((self.x-other.x)**2+(self.y-other.y)**2)**0.5
    def to_array(self) ->np.ndarray:
        return np.array([self.x,self.y])
    def to_vec3d(self) -> Vec3d:
        return Vec3d(self.x,self.y)
    def is_on_edge(self, edge:"Edge", include_endpoints:bool=True) ->bool:
        """点在曲线上"""
        return edge.touches_node(self,include_endpoints)

class Edge(Geom):  
    """边/曲线段"""
    def __init__(self,s:Node,e:Node) -> None:
        super().__init__()
        self.s,self.e=s,e
    @abstractmethod
    def equals(self,other:"Edge")->bool:...
    @abstractmethod
    def reverse(self) -> None:
        """原地反转。创建新的对象另见Edge.opposite()"""
        ...
    @abstractmethod
    def opposite(self) -> "Edge":
        """方向相反的新的边。原地反转另见Edge.reverse()"""
        ...
    @property
    @abstractmethod
    def length(self) -> float: ...
    def is_zero(self)->bool:
        return self.length<Const.TOL_DIST
    @abstractmethod
    def point_at(self,t:float) -> Node:
        """参数t处的点"""
        ...
    @abstractmethod
    def tangent_at(self,t:float)->Vec3d:
        """参数t处的单位切向量"""
        ...
    @abstractmethod
    def principal_normal_at(self,t:float)->Vec3d:
        """参数t处的单位主法向量"""
        ...
    def binormal_at(self,t:float)->Vec3d:
        """参数t处的单位副法向量"""
        return self.tangent_at(t).cross(self.principal_normal_at(t))
    @abstractmethod
    def radius_at(self,t:float,signed:bool=False)->float:
        """参数t处的曲率半径

        Args:
            t (float): 参数
            signed (bool, optional): 是否区分正负，符号同凸度. Defaults to False.

        Returns:
            float: 曲率半径
        """
        if signed: 
            return self.curvature_at(0)*self.binormal_at(0).z
        else:
            return abs(self.curvature_at(0)*self.binormal_at(0).z)
    @abstractmethod
    def curvature_at(self,t:float)->float: 
        """参数t处的曲率"""
        ...
    def frame_at(self,t:float)->Mat3d:
        """参数t处的frame: Columns of Mat3d"""
        vx=self.tangent_at(t)
        vy=self.principal_normal_at(t)
        vz=vx.cross(vy)
        return Mat3d.from_column_vecs([vx,vy,vz])
    @abstractmethod
    def slice_between(self,a:float|Node,b:float|Node,extend:bool=True)->Self:
        
        """返回a->b的切片.

        Args:
            a (Node|float): 起点.
            b (Node|float): 终点，满足t(a)<=t(b).
            extend (bool, optional): 允许向外延伸. Defaults to True.

        Returns:
            Edge: a->b的切片.t(a)>t(b)时返回None.
        """
        ...
    @abstractmethod
    def touches_node(self,point:Node,include_endpoints:bool=True)->bool:
        """点在边上"""
        ...
    @abstractmethod
    def is_parallel(self, other:"Edge") -> bool:
        """平行，含共线"""
        ...
    @abstractmethod
    def is_collinear(self, other:"Edge")->bool:
        """共线"""
        ...
    @abstractmethod
    def is_on_same_direction(self,other:"Edge")->bool:
        """同向"""
        ...
    @abstractmethod
    def intersects(self,other:"Edge",)->bool: 
        """相交，含端点相交，不含共线相交""" 
        ...
    @abstractmethod
    def intersection(self, other:"Edge") -> list[Node]:
        """交点，含端点相交，不含共线相交"""
        ...
    @abstractmethod
    def projection(self, pt:Node) -> Node:
        """点在边上的投影"""
        ...
    @abstractmethod
    def overlap(self,other:"Edge") -> list["Edge"]:
        """求共线重叠的部分"""
        ...
    @abstractmethod
    def offset(self,dist:float) -> "Edge":
        """左正右负"""
        ...
    @abstractmethod
    def closest_point(self,other:Node) -> Node:
        """点到边的最近点"""
        ...
    @abstractmethod
    def get_param(self,p:Node,arc_length:bool=False) -> float|None:
        """点在边上的参数.

        Args:
            p (Node): 点.
            arc_length (bool, optional): True返回[0,self.length]，False返回[0,1]. Defaults to False.

        Returns:
            float|None: 不在Edge上时返回None.
        """
        ...
    @staticmethod
    def intersection_extended(e1:"Edge",e2:"Edge")->list[Node]:
        """Edge所在线的直线/圆上的交点"""
        if isinstance(e1,LineSeg) and isinstance(e2,LineSeg): 
            pt=Edge.intersection_of_lines(e1,e2)
            return [pt] if pt is not None else []
        if isinstance(e1,Arc) and isinstance(e2,Arc): 
            return Edge.intersection_of_circles(e1,e2)
        if isinstance(e1,LineSeg) and isinstance(e2,Arc):
            return Edge.intersection_of_circle_and_line(e2,e1)
        if isinstance(e1,Arc) and isinstance(e2,LineSeg):
            return Edge.intersection_of_circle_and_line(e1,e2)
        raise TypeError("Unsupported type of geometry")
    @staticmethod
    def intersection_of_circles(arc1:"Arc",arc2:"Arc")->list[Node]:
        """两个圆弧所在的圆周求交，不含重合"""
        if not isinstance(arc1,Arc) and isinstance(arc2,Arc): raise TypeError()
        c1,r1=arc1.center,arc1.radius
        c2,r2=arc2.center,arc2.radius
        dst=c1.dist(c2)  # 圆心距
        comp=Const.cmp_dist
        if c1.equals(c2) and comp(r1,r2)==0:  # 重合
            return []
        if (comp(dst,r1+r2)>0 or comp(dst,abs(r1-r2))<0):  # 相离 or 包含
            return []
        if comp(dst,r1+r2)==0 or comp(dst,abs(r1-r2))==0:  # 外切 or 内切
            if r2>r2: c1,c2,r1,r2=c2,c1,r2,r1  # 确保内切时C1在外C2在内
            # 切点=C1+(C1->C2)*R1
            v=(c2.to_vec3d()-c1.to_vec3d()).unit()
            t_point=Node.from_vec3d(c1.to_vec3d()+v*r1)
            return [t_point,t_point]
        # else: 相交
        p=(r1+r2+dst)/2  # 海伦公式求两个圆心+任一交点构成的三角形面积
        s=(p*(p-r1)*(p-r2)*(p-dst))**0.5
        h=2*s/dst  # 交点投影到两个圆心连线的距离
        d=(r1**2-h**2)**0.5  # 圆心投影到两个交点连线的距离
        vd=(c2.to_vec3d()-c1.to_vec3d()).unit()
        cos1=(r1*r1+dst*dst-r2*r2)/(2*r1*dst)  # 余弦定理判断c1.center处的角度
        if cos1<0: vd=-vd  # 钝角三角形，vd在c1-c2的反向
        vh=vd.cross(Vec3d(0,0,1))
        return [Node.from_vec3d(c1.to_vec3d()+vd*d+vh*h),
                Node.from_vec3d(c1.to_vec3d()+vd*d-vh*h)]
    @staticmethod
    def intersection_of_circle_and_line(arc:"Arc",edge:"LineSeg") -> list[Node]:
        """圆弧所在的圆周与线段所在的直线求交"""
        if not isinstance(arc,Arc) or not isinstance(edge,LineSeg): raise TypeError
        res=[]
        projection=edge.projection(arc.center)
        h=projection.dist(arc.center)  # 弓高
        if h>arc.radius+Const.TOL_DIST:  # 圆和直线不交
            return res
        half_chord=abs(arc.radius**2-h**2)**0.5  # 半弦长        
        res=[Node.from_vec3d(projection.to_vec3d()+edge.to_vec3d().unit()*half_chord),
             Node.from_vec3d(projection.to_vec3d()-edge.to_vec3d().unit()*half_chord)]
        return res
    @staticmethod
    def intersection_of_lines(l1:"LineSeg",l2:"LineSeg")->Node:
        """求线段所在直线的交点"""
        if not isinstance(l1,LineSeg) or not isinstance(l2,LineSeg): raise TypeError
        if l1.is_parallel(l2): return None #平行（含共线）时无交点
        v1=(l1.e.y-l1.s.y,l1.s.x-l1.e.x,l1.e.x*l1.s.y-l1.s.x*l1.e.y)
        v2=(l2.e.y-l2.s.y,l2.s.x-l2.e.x,l2.e.x*l2.s.y-l2.s.x*l2.e.y)
        prod=(v1[1]*v2[2]-v2[1]*v1[2],-(v1[0]*v2[2]-v2[0]*v1[2]),v1[0]*v2[1]-v2[0]*v1[1])
        return Node(prod[0]/prod[2],prod[1]/prod[2])
    @staticmethod
    def compare_curvature_by_radius(a:float,b:float)->int:
        if (abs(a)==abs(b)==float("inf")
                or Const.cmp_dist(abs(a-b),0)==0): 
            return 0  # 直线
        if abs(a)==float("inf"): return (b<0)*2-1
        if abs(b)==float("inf"): return (a>0)*2-1
        return (a>b)*2-1 if a*b<0 else (a<b)*2-1
class Line(Geom):  # [TODO]: 把LineSeg直线相关的方法搬过来
    """直线"""
    def __init__(self,origin:Node,direction:Vec3d):
        super().__init__()
        self.origin=origin
        self.direction=direction.unit()
    def get_mbb(self):
        if abs(self.angle-math.pi/2)<Const.TOL_ANG: 
            x1=x2=self.origin.x
        else:
            x1,x2=-math.inf,math.inf
        if self.angle<Const.TOL_ANG: 
            y1=y2=self.origin.y
        else: 
            y1,y2=-math.inf,math.inf
        return (Node(x1,y1),Node(x2,y2))
    @property
    def angle(self)->float:
        """直线的角度, 范围[0,pi), 含误差"""
        angle=self.direction.angle
        if angle>=math.pi: angle-=math.pi
        if math.pi-angle<Const.TOL_ANG: angle-=math.pi
        return angle
    def is_parallel(self,other):
        ...
class Ray(Geom):  # [TODO]: 实现LineSeg相关的方法
    """射线"""
    def __init__(self,origin:Node,direction:Vec3d):
        super().__init__()
        self.origin=origin
        self.direction=direction.unit()
    def get_mbb(self):
        a=self.angle
        if a<Const.TOL_ANG or abs(a-math.pi)<Const.TOL_ANG:  # ↔
            y1=y2=self.origin.y
        elif a>math.pi:  # ⬇
            y1,y2=-math.inf,self.origin
        else:  # ⬆
            y1,y2=self.origin.y,math.inf
        if abs(a-math.pi/2)<Const.TOL_ANG or abs(a-math.pi/2*3)<Const.TOL_ANG:  # ↕
            x1=x2=self.origin.x
        elif math.pi/2<a<math.pi/2*3:  # ⬅
            x1,x2=-math.inf,self.origin.x
        else:  # ➡
            x1,x2=self.origin.x,math.inf
        return (Node(x1,y1),Node(x2,y2))
    @property
    def angle(self)->float:
        """射线的角度, 范围[0,2pi), 含误差"""
        return self.direction.angle
class LineSeg(Edge):
    """直线段"""
    def __init__(self, s:Node, e:Node) -> None:
        super().__init__(s,e)
    def __repr__(self):
        return f"LineSeg({self.s},{self.e})"
    def __eq__(self,other):
        return self.equals(other)
    def __hash__(self)->int:
        return id(self)
    def equals(self,other:"LineSeg")->bool:
        return isinstance(other,LineSeg) and self.s==other.s and self.e==other.e
    @classmethod
    def from_origin_direction_length(cls,origin:Node,direction:Vec3d,length:float)->"LineSeg":
        return cls(origin,origin+direction.unit()*length)
    @classmethod
    def from_array(cls,arr:np.ndarray) -> "LineSeg":
        if arr.shape==(2,2):
            return cls(Node.from_array(arr[0]),Node.from_array(arr[1]))
        else: return None
    def get_mbb(self) -> tuple["Node", "Node"]:
        return (Node(min(self.s.x,self.e.x),min(self.s.y,self.e.y)),Node(max(self.s.x,self.e.x),max(self.s.y,self.e.y)))
    def reverse(self) -> None:
        self.s,self.e=self.e,self.s
    def opposite(self) -> "LineSeg":
        return LineSeg(self.e,self.s)
    @property
    def length(self) -> float:
        return self.s.dist(self.e)
    @property
    def angle(self) -> float:
        """角度，范围[0,2pi), 含误差"""
        return self.to_vec3d().angle
    @property
    def angle_of_line(self)->float:
        """求线段所在直线的角度, 范围[0,pi), 含误差"""
        self.to_line().angle
    @property
    def coefficients(self) -> tuple[float,float,float]:
        """求线段所在直线方程ax+by+c=0的系数"""
        if self.s.equals(self.e): return 0,0,0
        return self.s.y-self.e.y,self.e.x-self.s.x,self.s.x*self.e.y-self.e.x*self.s.y
    def to_vector_array(self) -> np.ndarray:
        return self.e.to_array()-self.s.to_array()
    def to_array(self) -> np.ndarray:
        return np.array([self.s.to_array(),self.e.to_array()])
    def to_vec3d(self) -> Vec3d:
        return Vec3d(self.e.x,self.e.y)-Vec3d(self.s.x,self.s.y)
    def to_line(self) -> Line:
        return Line(self.s,self.to_vec3d())
    def is_point_on_line(self,point:Node)->bool:
        """点在线段所在的直线上"""
        # 点到直线的投影距离=0
        return point.dist(self.projection(point))<Const.TOL_DIST
    def touches_node(self,point:Node,include_endpoints:bool=True)->bool:
        """点在线段上"""
        if self.is_zero(): return self.s.equals(point)
        # 点是否在直线上
        if not self.is_point_on_line(point): return False
        # 点是否在端点上
        if point.dist(self.s)<Const.TOL_DIST or point.dist(self.e)<Const.TOL_DIST:
            return include_endpoints
        # 点在线段内
        return 0<self.get_param(point)<1
    def is_parallel(self, other:Edge) -> bool:
        """平行，含共线；认为点和任意曲线都平行"""
        if self.is_zero() or other.is_zero(): return True
        if not isinstance(other,LineSeg):
            # raise TypeError('The other object must be LineSeg.')
            return False
            # return abs(other.bulge)<Const.TOL_VAL and self.is_parallel(LineSeg(other.s,other.e))
        if isinstance(other,LineSeg):
            # 四个点互相投影，判断平行距离相等；这样会拖慢速度
            # v=[other.projection(self.s).to_vec3d()-self.s.to_vec3d(),
            #    other.projection(self.e).to_vec3d()-self.e.to_vec3d(),
            #    -(self.projection(other.s).to_vec3d()-other.s.to_vec3d()),
            #    -(self.projection(other.e).to_vec3d()-other.e.to_vec3d()),
            #    ]
            # for i in range(3):
            #     for j in range(i+1,4):
            #         if not v[i].equals(v[j]):
            #             return False
            # 丑而快的写法
            vself=self.to_vec3d().unit()
            vos=other.s.to_vec3d()-self.s.to_vec3d()
            v1=vos-vself*vos.dot(vself)
            voe=other.e.to_vec3d()-self.s.to_vec3d()
            v2=voe-vself*voe.dot(vself)
            if not v1.equals(v2): return False
            vother=other.to_vec3d().unit()
            vss=self.s.to_vec3d()-other.s.to_vec3d()
            v3=-(vss-vother*vss.dot(vother))
            if not v1.equals(v3): return False
            if not v2.equals(v3): return False
            vse=self.e.to_vec3d()-other.s.to_vec3d()
            v4=-(vse-vother*vse.dot(vother))
            if not v1.equals(v4): return False
            if not v2.equals(v4): return False
            if not v3.equals(v4): return False
            return True
        return False
    def is_collinear(self, other:Edge)->bool:
        """共线"""
        if not self.is_parallel(other): return False
        if isinstance(other,Arc):
            return False
            # return abs(other.bulge)<Const.TOL_VAL and self.is_collinear(LineSeg(other.s,other.e))
        if isinstance(other,LineSeg):
            # 判断各端点在另一条直线上的投影距离是0
            cond1=self.s.dist(other.projection(self.s))<Const.TOL_DIST
            cond2=self.e.dist(other.projection(self.e))<Const.TOL_DIST
            cond3=other.s.dist(self.projection(other.s))<Const.TOL_DIST
            cond4=other.e.dist(self.projection(other.e))<Const.TOL_DIST
            return cond1 and cond2 and cond3 and cond4
    def is_on_same_direction(self,other:Edge)->bool:
        """线段同向"""
        if not isinstance(other,LineSeg): return False
        if self.is_zero() or other.is_zero(): return True  # 零线段和所有人都同向
        return (self.is_parallel(other)
                and self.to_vec3d().dot(other.to_vec3d())>Const.TOL_VAL)
    def point_at(self,t:float) -> Node:
        return Node(self.s.x+(self.e.x-self.s.x)*t,self.s.y+(self.e.y-self.s.y)*t)
    def tangent_at(self,t:float)->Vec3d:
        return self.to_vec3d().unit()
    def principal_normal_at(self,t:float)->Vec3d:
        return self.to_vec3d().unit().rotate2d(math.pi/2)
    def radius_at(self,t:float,signed:bool=False)->float:
        return float("inf")
    def curvature_at(self,t:float,signed:bool=False)->float: 
        return 0
    def angle_to(self,other:"LineSeg")->float:
        """求线段到other的旋转角[0,2pi)"""
        return self.to_vec3d().angle_to(other.to_vec3d())
    def intersects(self,other:Edge)->bool:
        """相交，含端点相交，不含共线相交"""
        if self.is_parallel(other):return False
        intersections=self.intersection(other)
        for p in intersections:
            if self.touches_node(p) and other.touches_node(p):
                return True
        return False
    def intersection(self, other:Edge) -> list[Node]: 
        """求线段与Edge的交点"""
        if isinstance(other,Arc): return other.intersection(self)
        if isinstance(other,LineSeg):
            p=Edge.intersection_of_lines(self,other)
            if p is not None and self.touches_node(p) and other.touches_node(p):
                return [p]
            else: return []
    def get_param(self,p:Node,arc_length:bool=False) -> float|None:
        """点在边上的参数.

        Args:
            p (Node): 点.
            arc_length (bool, optional): True返回[0..self.length]，False返回[0..1]. Defaults to False.

        Returns:
            float|None: 不在Edge上时返回None.
        """
        l=self.length
        if self.s.equals(p): return 0
        if self.e.equals(p): return l if arc_length else 1
        v1=(self.e.x-self.s.x,self.e.y-self.s.y)
        v2=(p.x-self.s.x,p.y-self.s.y)
        t=(v1[0]*v2[0]+v1[1]*v2[1])/l
        return t if arc_length else t/l
    def projection(self, pt:Node) -> Node:
        """求pt在self所在直线上的投影点"""
        if self.is_zero(): return self.point_at(0.5)
        vp=pt.to_vec3d()-self.s.to_vec3d()
        v0=self.to_vec3d().unit()
        vtrans=v0*vp.dot(v0)
        vproj=self.s.to_vec3d()+vtrans
        return Node(vproj.x,vproj.y)
    def overlap(self,other:"LineSeg") -> list["LineSeg"]:
        """求线段的重叠部分"""
        if not self.is_collinear(other): return [] #不共线则不重叠
        pMin1,pMax1=self.s,self.e
        pMin2,pMax2=other.s,other.e
        proj=lambda pt:pt.x*(self.e.x-self.s.x)+pt.y*(self.e.y-self.s.y) #点投影到self向量的函数
        min1,max1,min2,max2=map(proj,(pMin1,pMax1,pMin2,pMax2)) #4个端点都投影到self向量上
        if min2>max2: #排序
            min2, max2=max2, min2
            pMin2, pMax2=pMax2, pMin2
        if min2>max1+Const.TOL_DIST or min1>max2+Const.TOL_DIST: #没有重叠的情况返回None
            return []
        pMin=pMin1 if min1>min2 else pMin2 #左侧重叠点
        pMax=pMax1 if max1<max2 else pMax2 #右侧重叠点
        return [LineSeg(pMin,pMax)]
    def closest_point(self,other:Node) -> Node:
        """求点到线段的最近点"""
        dot_prod=(other.x-self.s.x)*(self.e.x-self.s.x)+(other.y-self.s.y)*(self.e.y-self.s.y)
        t=dot_prod/(self.length**2)
        t=max(min(t,1),0)
        return self.point_at(t)
    def slice_between(self,a:float|Node,b:float|Node)->Self:
        t1=a if isinstance(a,(int,float)) else self.get_param(a)
        t2=b if isinstance(b,(int,float)) else self.get_param(b)
        p1=a if isinstance(a,Node) else self.point_at(a)
        p2=b if isinstance(b,Node) else self.point_at(b)
        if t1<t2 or p1.equals(p2): 
            return LineSeg(p1,p2)
        else: return None
    def offset(self,dist:float) -> "LineSeg": 
        """左正右负"""
        vector=(self.e.x-self.s.x,self.e.y-self.s.y)
        left_unit=(-vector[1]/self.length,vector[0]/self.length)
        newS=Node(self.s.x+left_unit[0]*dist,self.s.y+left_unit[1]*dist)
        newE=Node(self.e.x+left_unit[0]*dist,self.e.y+left_unit[1]*dist)
        return LineSeg(newS,newE)
    def fillet_with(self,other:"LineSeg",radius:float)->"Arc":
        """fillet with next edge, 有方向"""
        if self.is_collinear(other):
            new_edge=LineSeg(self.e,other.s)
            return Arc(self.e,self.s,0)
        elif self.is_parallel(other):
            ...
        else:
            v1=self.to_vec3d()
            v2=other.to_vec3d()
            # offset
            if v1.cross(v2).z>0: #左转
                l1=self.offset(radius)
                l2=other.offset(radius)
            else: #右转
                l1=self.offset(-radius)
                l2=other.offset(-radius)
            # 圆心
            c=l1.intersection(l2)
            # 切点
            p1=self.projection(c)
            p2=other.projection(c)
            # 圆心->切点向量
            vr1=p1.to_vec3d()-c.to_vec3d()
            vr2=p2.to_vec3d()-c.to_vec3d()
            # 圆心角 = 夹角, (-pi,pi]
            angle=self.angle_to(other)
            if angle>math.pi: angle=angle-2*math.pi
            arc=Arc(p1,p2,math.tan(angle/4))
            return arc
class Arc(Edge):
    """圆弧"""
    def __init__(self,s:Node,e:Node,bulge:float) -> None:
        """从起点、终点、凸度构造圆弧

        Args:
            s (Node): 起点
            e (Node): 终点
            bulge (float): 凸度（>0逆时针，弧在弦右边；<0顺时针，弧在弦左边）
        """
        super().__init__(s,e)
        self.bulge=bulge
    def __repr__(self) -> str:
        return f"Arc({self.s},{self.e},{self.bulge})"
    def __eq__(self,other):
        return self.equals(other)
    def __hash__(self) ->int:
        return id(self)
    def equals(self,other:"Arc")->bool:
        return isinstance(other,Arc) and self.s==other.s and self.e==other.e and abs(self.bow_height-other.bow_height)<Const.TOL_DIST
    @property
    def bow_height(self)->float:
        """弓高，分正负"""
        return self.s.dist(self.e)/2*self.bulge
    @classmethod
    def from_center_radius_angle(cls,center_point:Node,radius:float,start_angle:float,total_angle:float)->"Arc":
        """从圆心、半径、起始角度、总角度构造圆弧

        Args:
            center_point (Node): 圆心
            radius (float): 半径
            start_angle (float): 起始角度
            total_angle (float): 总角度（>0逆时针；<0顺时针）

        Returns:
            Arc: 圆弧
        """
        if total_angle<0:
            start_angle,end_angle=end_angle,start_angle
            total_angle=-total_angle
        end_angle=start_angle+total_angle
        if end_angle+Const.TOL_ANG>math.pi*2:
            end_angle-=math.pi*2
        s=Node(center_point.x+radius*math.cos(start_angle),
               center_point.y+radius*math.sin(start_angle))
        e=Node(center_point.x+radius*math.cos(end_angle),
               center_point.y+radius*math.sin(end_angle))
        bulge=math.tan(total_angle/4)
        # if abs(bulge)<Const.TOL_VAL: return LineSeg(s,e)
        # else: return cls(s,e,bulge)
        return cls(s,e,bulge)
    def get_mbb(self) -> tuple[Node, Node]:
        x,y,z,r=self.center.x,self.center.y,self.center.z,self.radius
        left,right,top,bottom=Node(x-r,y,z),Node(x+r,y,z),Node(x,y+r,z),Node(x,y-r,z)
        pmin_x=left.x if self.touches_node(left) else min(self.s.x,self.e.x)
        pmin_y=bottom.y if self.touches_node(bottom) else min(self.s.y,self.e.y)
        pmax_x=right.x if self.touches_node(right) else max(self.s.x,self.e.x)
        pmax_y=top.y if self.touches_node(top) else max(self.s.y,self.e.y)
        return (Node(pmin_x,pmin_y),Node(pmax_x,pmax_y))
    def reverse(self)->None:
        self.s,self.e,self.bulge=self.e,self.s,-self.bulge
    def opposite(self) -> "Arc":
        return Arc(self.e,self.s,-self.bulge)
    @property
    def center(self)->Node:
        b=0.5*(1/self.bulge-self.bulge)
        x=0.5*((self.s.x+self.e.x)-b*(self.e.y-self.s.y))
        y=0.5*((self.s.y+self.e.y)+b*(self.e.x-self.s.x))
        return Node(x,y)
    @property
    def radius(self)->float:
        b=0.5*(1/self.bulge+self.bulge)
        l=self.s.dist(self.e)
        return abs(l/2*b)
    @property
    def angles(self)->tuple[float,float]:
        edge1=LineSeg(self.center,self.s)
        edge2=LineSeg(self.center,self.e)
        return (edge1.angle,edge2.angle)
    @property
    def radian(self)->float:
        return math.atan(self.bulge)*4
    @property
    def length(self)->float:
        if abs(self.bulge)<Const.TOL_VAL: return self.s.dist(self.e)
        return abs(self.radius*self.radian)
    def is_point_on_circle(self,point:Node)->bool:
        """点在圆弧所在的圆周上"""
        if self.is_zero(): return self.s.equals(point)
        return abs(point.dist(self.center)-self.radius)<Const.TOL_DIST
    def touches_node(self,point:Node,include_endpoints:bool=True)->bool:
        """点在圆弧上"""
        # 点是否在圆周上
        if not self.is_point_on_circle(point): return False
        # 点是否在端点上
        if point.dist(self.s)<Const.TOL_DIST or point.dist(self.e)<Const.TOL_DIST:
            return include_endpoints
        # 点在线段内
        return 0<self.get_param(point)<1
    def is_parallel(self, other:Edge) -> bool:
        """圆弧平行"""
        if self.is_zero() or other.is_zero(): return True
        if not isinstance(other,Arc): return False
        return self.center.equals(other.center)
    def is_collinear(self, other:Edge)->bool:
        """圆弧共圆"""
        if not isinstance(other,Arc): return False
        # 两个点是否定义为共圆待定 TODO
        if self.is_zero() and other.is_zero(): return True
        # 其中一个是点，则它需要在另一个圆周上
        if self.is_zero() and other.is_point_on_circle(self.s) and other.is_point_on_circle(self.e): return True
        if other.is_zero() and self.is_point_on_circle(other.s) and self.is_point_on_circle(other.e): return True
        return self.is_parallel(other) and abs(self.radius-other.radius)<Const.TOL_DIST
    def is_on_same_direction(self,other:Edge)->bool:
        """圆弧同向"""
        if not isinstance(other,Arc): return False
        if self.is_zero() or other.is_zero(): return True  # 零线段和所有人都同向        
        return (abs(self.radius-other.radius)<Const.TOL_DIST
                and (self.bulge>0)==(other.bulge>0))
    def closest_point(self, other: Node) -> Node:
        if other.equals(self.center): return self.s
        edge=Edge(self.center,other)
        possible_intersections=Edge.intersection_of_circle_and_line(self,edge)
        for p in possible_intersections:
            v1=other.to_vec3d()-self.center.to_vec3d()
            v2=p.to_vec3d()-self.center.to_vec3d()
            if v1.dot(v2)>0 and self.touches_node(p):  # 交点在圆弧角度范围内返回交点
                return p
        return self.s if self.s.dist(other)<self.e.dist(other) else self.e  # 交点不在圆弧角度范围内返回端点
    def intersects(self,other:Edge)->bool:
        if self.is_parallel(other):return False
        return len(self.intersection(other))>0
    def intersection(self,other:Edge) -> list[Node]:
        if self.is_parallel(other): return []
        if isinstance(other,LineSeg): return self._intersection_with_line_segment(other)
        if isinstance(other,Arc): return self._intersection_with_arc(other)
    def _intersection_with_arc(self,other:"Arc") -> list[Node]:
        """圆弧与圆弧求交"""
        res=[]
        possible_intersections=Edge.intersection_of_circles(self,other)
        for p in possible_intersections:
            if self.touches_node(p) and other.touches_node(p): 
                res.append(p)
        return res
    def _intersection_with_line_segment(self,other:LineSeg) -> list[Node]:
        """圆弧与线段求交"""
        res=[]
        possible_intersections=Edge.intersection_of_circle_and_line(self,other)
        for p in possible_intersections:
            if self.touches_node(p) and other.touches_node(p):
                res.append(p)
        return res
    def get_param(self,p:Node,arc_length:bool=False) -> float|None:
        """点在边上的参数.

        Args:
            p (Node): 点.
            arc_length (bool, optional): True返回[0..self.length]，False返回[0..1]. Defaults to False.

        Returns:
            float|None: 不在Edge上时返回None.
        """
        l=self.length
        if self.s.equals(p): return 0
        if self.e.equals(p): return l if arc_length else 1
        if self.is_zero() or not self.is_point_on_circle(p): return None
        v_p=p.to_vec3d()-self.center.to_vec3d()
        v_s=self.s.to_vec3d()-self.center.to_vec3d()
        radian_s2p=v_s.angle_to(v_p)
        if self.bulge<0 and abs(radian_s2p)>Const.TOL_ANG: radian_s2p=radian_s2p-2*math.pi
        t=radian_s2p/self.radian
        return t*l if arc_length else t
    def point_at(self,t:float) -> Node:
        t_range=2*math.pi/self.radian  # 圆周的t的范围
        t=t%t_range
        angle=self.angles[0]+t*self.radian  # 点的角度
        return Node(self.center.x+self.radius*math.cos(angle),self.center.y+self.radius*math.sin(angle))
    def tangent_at(self,t:float)->Vec3d:
        t_range=2*math.pi/self.radian  # 圆周的t的范围
        angle=self.angles[0]+t*self.radian  # 点的角度
        vec=Vec3d(math.cos(angle),math.sin(angle))
        tangent=Vec3d(0,0,1).cross(vec) if self.bulge>0 else vec.cross(Vec3d(0,0,1))
        return tangent
    def principal_normal_at(self,t:float)->Vec3d:
        if self.bulge>Const.TOL_VAL:
            return self.tangent_at(t).rotate2d(math.pi/2)
        else: 
            return self.tangent_at(t).rotate2d(-math.pi/2)
    def radius_at(self,t:float,signed:bool=False)->float:
        if signed and self.bulge<0: 
            return -self.radius
        else:
            return self.radius
    def curvature_at(self,t:float,signed:bool=False)->float: 
        if signed and self.bulge<0:
            return -1/self.radius
        else:
             return 1/self.radius
    def overlap(self,other:Edge) -> list["Arc"]:
        if not self.is_collinear(other): return [] #不共线则不重叠
        # 先把圆弧的方向都变成逆时针
        arc1=self if self.bulge>0 else self.opposite()
        arc2=other if other.bulge>0 else other.opposite()
        # 投影到self上
        t_s1,t_e1=0,1
        t_s2,t_e2=arc1.get_param(arc2.s), arc1.get_param(arc2.e)
        res=[]
        if t_s1<t_e1<t_e2<t_s2:  # arc2包含arc1：返回arc1
            return [arc1]
        elif t_s1<t_s2<t_e2<t_e1:  # arc1包含arc2：返回arc2
            return [arc2]
        elif t_s1<t_e2<t_e1<t_s2:  # arc2后半部分和arc1前半部分重叠
            return [self.slice_between(arc1.s,arc2.e)]
        elif t_s1<t_s2<t_e1<t_e2:  # arc2前半部分和arc1后半部分重叠
            return [self.slice_between(arc2.s,arc1.e)]
        elif t_s1<t_e2<t_s2<t_e1:  # arc2和arc1有两段重叠
            return [self.slice_between(arc1.s,arc2.e),self.slice_between(arc2.s,arc1.e)]
        else:  # t_s1<t_e1<t_s2<t_e2:  # arc2和arc1不重叠
            return []
    def projection(self, other:Node) -> Node:
        """求pt在self所在圆周上的最近投影点"""
        if other.equals(self.center): return self.s
        edge=Edge(self.center,other)
        possible_projections=Edge.intersection_of_circle_and_line(self,edge)
        dists=[other.dist(p) for p in possible_projections]
        nearest_p=possible_projections[dists.index(min(dists))]
        return nearest_p
    def slice_between(self,a:float|Node,b:float|Node)->Self:
        t1=a if isinstance(a,(int,float)) else self.get_param(a)
        t2=b if isinstance(b,(int,float)) else self.get_param(b)
        p1=a if isinstance(a,Node) else self.point_at(a)
        p2=b if isinstance(b,Node) else self.point_at(b)
        radian=self.radian*(t2-t1)
        arc_length=self.radius*radian
        if t1<t2 or arc_length<Const.TOL_DIST:
            return Arc(p1,p2,math.tan(radian/4))
        else: return None
    def fit(self,quad_segs:int=16,min_segs:int=1) -> list[LineSeg]:
        subdiv_num=max(min_segs,math.ceil(abs(self.radian/(math.pi/2)*quad_segs)))
        if subdiv_num==0: return [LineSeg(self.s,self.e)]
        subdiv_radian=self.radian/subdiv_num
        nodes=[self.s]
        for i in range(1,subdiv_num):
            inter_angle=self.angles[0]+subdiv_radian*i
            v=Vec3d(math.cos(inter_angle),math.sin(inter_angle))*self.radius
            nodes.append(Node(self.center.x+v.x,self.center.y+v.y))
        nodes.append(self.e)
        return [LineSeg(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]    
    def offset(self,dist:float,cross_center:bool=False) -> "Arc":
        """偏移

        Args:
            dist (float): 左正右负
            cross_center (bool, optional): dist>半径时允许穿越圆心. Defaults to False.

        Returns:
            Arc: 偏移后的
        """
        if abs(self.bulge)<Const.TOL_VAL: return LineSeg(self.s,self.e,dist)
        if self.bulge>0: dist=-dist  # 圆心在弦的左侧，此时向左偏意味着半径减小
        if self.radius+dist<Const.TOL_DIST and not cross_center:  # 圆心穿越
            return LineSeg(self.center,self.center)
        vs=(self.s.to_vec3d()-self.center.to_vec3d()).unit()
        ve=(self.e.to_vec3d()-self.center.to_vec3d()).unit()
        new_s=Node.from_vec3d(self.s.to_vec3d()+vs*dist)
        new_e=Node.from_vec3d(self.e.to_vec3d()+ve*dist)
        return Arc(new_s,new_e,self.bulge)
class Circle(Arc):
    """圆周"""
    _dumper_ignore=["radian","bulge","angles"]
    def __init__(self,center:Node,radius:float) -> None:
        s=Node(center.x+radius,center.y)
        super().__init__(s,s,math.inf)
        self._center=center
        self._radius=radius
        self._radian=math.pi*2
        self._angles=(0,math.pi*2)
    @property
    def center(self) -> Node:
        return self._center
    @property
    def radius(self) -> Node:
        return self._radius
    @property
    def radian(self) -> Node:
        return self._radian    
    @property
    def angles(self) -> Node:
        return self._angles 
    @classmethod
    def from_center_start(cls,center:Node,start_point:Node)->Self:
        new_ins=cls(center,center.dist(start_point))
        new_ins.s=new_ins.e=start_point
        angle=(start_point.to_vec3d()-center.to_vec3d()).angle
        new_ins.angles=(angle,angle+math.pi*2)
    def __repr__(self) -> str:
        return f"Circle({self.center},{self.radius})"
    def __eq__(self,other):
        return self.equals(other)
    def __hash__(self) ->int:
        return id(self)
    def __copy__(self)->Self:
        return Circle(self.center,self.radius)
    def __deepcopy__(self)->Self:
        return Circle(deepcopy(self.center),self.radius)
    def get_mbb(self):
        return (Node(self.center.x-self.radius,self.center.y-self.radius),
                Node(self.center.x+self.radius,self.center.y+self.radius))
    def equals(self,other:"Circle")->bool:
        return isinstance(other,Circle) and self.center.dist(other.center)+abs(self.radius-other.radius)<Const.TOL_DIST
    def to_halves(self)->list["Arc"]:
        return [Arc.from_center_radius_angle(self.center,self.radius,0,math.pi),
                Arc.from_center_radius_angle(self.center,self.radius,math.pi,math.pi)]
        
class Polyedge(Geom):
    """多段线"""
    def __init__(self,nodes:list[Node],bulges:list[float]=None):
        """从顶点+凸度构造多段线

        Args:
            nodes (list[Node]): 顶点
            bulges (list[float]): 后一条边的凸度
        """
        bulges=bulges or [0]*len(nodes)
        if not (len(nodes)>=2 and len(nodes)==len(bulges)): 
            raise ValueError("Node/bulge numbers not matching.")
        self.nodes:list[Node]=nodes[:]
        self.bulges:list[int]=bulges[:]
        # edges不是intrinsic属性，现场计算比较好
        # self.edges:list[Edge]=[]
        # for i in range(len(self.nodes)-1):
        #     s,e,bulge=self.nodes[i],self.nodes[i+1],nodes[i][1]
        #     self.edges.append(LineSeg(s,e) if bulge==0 else Arc(s,e,bulge))
    def __len__(self)->int: return len(self.nodes)if self.is_closed else len(self.nodes)-1
    @property
    def is_closed(self)->bool:return False
    def __eq__(self,other):
        return self.equals(other)
    def __hash__(self)->int:
        return id(self)
    def equal(self,other:"Polyedge")->bool:
        return isinstance(other,Polyedge) and self.nodes==other.nodes and self.bulges==self.bulges
    @classmethod
    def from_edges(cls,edges:list[LineSeg|Arc]) -> "Polyedge":
        """从边构造多段线

        Args:
            edges (list[LineSeg | Arc]): 要求依次严格首尾相连(s is e)
        """
        for i in edges:
            if not (i==0 or edges[i].s is edges[i-1].e): raise ValueError("Edges not continuous.")
        nodes=[edge.s for edge in edges]+[edges[-1].e]
        bulges=[edge.bulge if isinstance(edge,Arc) else 0 for edge in edges]+[0]
        return cls([tup for tup in zip(nodes,bulges)])
    def close(self)->"Loop":
        return Loop(list(zip(self.nodes,self.bulges)))
    def get_mbb(self) -> tuple[Node, Node]:
        return Geom.merge_mbb([e.get_mbb() for e in self.edges])
    def to_array(self) -> np.ndarray:
        if self.is_closed:
            return np.array([node.to_array() for node in self.nodes]+[self.nodes[0].to_array()])
        else:
            return np.array([node.to_array() for node in self.nodes])
    @property
    def xy(self) ->np.ndarray:
        return self.to_array().T    
    def edge(self,index:int)->Edge:
        """获取第index条边

        Args:
            index (int): index from (-len) to (len-1), where len=edge_count=node_count-1

        Returns:
            Edge: 第index条边(现场计算的一个new instance)
        """
        if not (-len(self)<=index<len(self)): raise IndexError("Index out of range.")
        if index<0: index+=len(self)
        s,e,bulge=self.nodes[index],self.nodes[(index+1)%len(self.nodes)],self.bulges[index]
        return LineSeg(s,e) if bulge==0 else Arc(s,e,bulge)
    @property
    def edges(self)->Generator[Edge,None,None]:
        for i in range(len(self)):
            yield self.edge(i)
class Loop(Polyedge):
    """环(Closed PolyEdge)"""
    _dumper_ignore=["prepared"]
    def __init__(self,nodes:list[Node],bulges:list[float]=None,prepare:bool=False):
        """从顶点+凸度构造环.

        Args:
            nodes (list[Node]): 顶点.
            bulges (list[float]): 后一条边的凸度.
            prepare (bool, optional): 是否构造边集的搜索树. Defaults to True.
        """
        super().__init__(nodes,bulges)
        self.prepared=None
        if prepare: self.prepare()
    def __len__(self)->int: 
        return len(self.nodes)
    def __copy__(self)->Self:
        return Loop(self.nodes,self.bulges)
    def __deepcopy__(self)->Self:
        return Loop(deepcopy(self.nodes),self.bulges)
    @property
    def is_closed(self)->bool:return True
    @classmethod
    def from_edges(cls,edges:list[LineSeg|Arc],prepare:bool=False) -> "Loop":
        """从边构造环

        Args:
            edges (list[LineSeg | Arc]): 要求依次严格首尾相连(s is e)
        """
        for i in range(len(edges)):
            if edges[i].s is not edges[i-1].e: raise ValueError("Edges not continuous.")
        nodes=[edge.s for edge in edges]
        bulges=[edge.bulge if isinstance(edge,Arc) else 0 for edge in edges]
        return cls(nodes,bulges,prepare=prepare)
    def prepare(self)->None:  # [TODO]: 替换求交算法中的引用
        # if self.prepared is not None: return
        self.prepared=STRTree(list(self.edges))
    def is_identical(self,other:"Loop")->bool:
        return abs(self.area-other.area)<Const.TOL_AREA and self.covers(other) and other.covers(self)
    def reverse(self) -> None:
        self.nodes.reverse()
        self.nodes=self.nodes[-1:]+self.nodes[:-1]  # 保持起点不变
        self.bulges.reverse()
    def reversed(self) -> Self:
        return Loop(self.nodes[0:1]+self.nodes[-1:0:-1],self.bulges[::-1])
    @property
    def length(self)->float:
        if hasattr(self,"_length"): return self._length
        self._length=sum([edge.length for edge in self.edges])
        return self._length
    @property
    def area(self) -> float:
        if hasattr(self,"_area"): return self._area
        self._area=0
        for edge in self.edges:
            bow_area=0
            if isinstance(edge,Arc) and not edge.is_zero():
                vs=edge.s.to_vec3d()-edge.center.to_vec3d()
                ve=edge.e.to_vec3d()-edge.center.to_vec3d()
                bow_area=edge.radian/2*edge.radius**2-0.5*(vs.cross(ve).dot(Vec3d.Z))
            self._area+=(edge.s.x*edge.e.y-edge.s.y*edge.e.x)/2+bow_area
        return self._area
    def get_centroid(self)->Node:  # 需测试 TODO
        """重心"""
        # 划分成三角形然后重心加权
        centroid=Vec3d(0,0)
        nodes=self.nodes
        p0=nodes[0].to_vec3d()
        for i in range(1,len(nodes)-1):
            p1,p2=nodes[i].to_vec3d(),nodes[i+1].to_vec3d()
            v01,v02=p1-p0,p2-p0
            triangle_area=0.5*v01.cross(v02).z
            triangle_centroid=(p0+p1+p2)/3
            centroid+=triangle_centroid*triangle_area/self.area
        return Node(centroid.x,centroid.y)
    def simplify(self,cull_dup=True,cull_insig=True)->None:  # TODO
        """去除环上冗余的顶点/边

        Args:
            cull_dup (bool, optional): 去除重复的顶点（长度为0的边）. Defaults to True.
            cull_insig (bool, optional): 去除共线的边中间的顶点. Defaults to True.
        """
        if cull_dup:
            cond=[False if edge.is_zero() else True for edge in self.edges]
            self.nodes=[node for i,node in enumerate(self.nodes) if cond[i] or i==len(self.nodes)-1]
            self.bulges=[bulge for bulge in self.bulges if cond[i]]
        if cull_insig:
            for start_i,start_edge in enumerate(self.edges):  # 先确定一个转折点的边作为起始边(start_i和start_i-1构成一个转折)
                if not start_edge.is_collinear(self.edges[start_i-1]):
                    break
            new_edges=[start_edge]
            i=start_i+1
            for _ in range(len(self.edges)):
                if not self.edges[i-1].is_on_same_direction(self.edges[i]):
                    new_edges.append(self.edges[i])
                else: 
                    new_edges[-1]=self.edges[i-1].slice_between_points(self.edges[i-1].s,self.edges[i].e,extend=True)
            
        self.update(update_node=True)
    def offset(self,side:str="left",dist:float=None,split:bool=True,mitre_limit:float=None) -> list["Loop"]:
        def comb_dist(edge:Edge,dist:float)->float:
            """换算左右距离"""
            if dist is None:
                lw=edge.lw if hasattr(edge,"lw") else 0
                rw=edge.rw if hasattr(edge,"rw") else 0                
                return -rw if side=="right" else lw
            else: 
                return -dist if side=="right" else dist
        def meets_miter_limit(pre_edge_offset:Edge,edge_offset:Edge,intersection:Node,mitre_limit:float)->bool:
            if pre_edge_offset.get_param(intersection)>=0 and edge_offset.get_param(intersection)<=1:  # 交点在前一段的起点之后、后一段的终点之前
                return True
            if pre_edge_offset.get_param(intersection)<0 and intersection.dist(pre_edge_offset.s)<mitre_limit:  # 交点在前一段的起点之前，但是没有超过限制
                return True
            if edge_offset.get_param(intersection)>1 and intersection.dist(edge_offset.e)<mitre_limit:  # 交点在后一段的终点之后，但是没有超过限制
                return True
            return False
        if mitre_limit is None: mitre_limit=Const.MAX_VAL
        new_nodes=[]
        pre_edge=self.edges[-1]
        pre_edge_offset=pre_edge.offset(comb_dist(pre_edge,dist))
        for edge in self.edges:
            edge_offset=edge.offset(comb_dist(edge,dist))
            # 第一种情况：相交，包括直接相交或者延长后交上 TODO
            intersection=Edge.intersection_extended(pre_edge_offset,edge_offset)
            if len(intersection)>0 and meets_miter_limit(pre_edge_offset,edge_offset,intersection[0],mitre_limit):
                new_nodes.append(intersection[0])
            # 第二种情况：重叠，包括点重叠 TODO                
            elif len(overlap:=pre_edge_offset.overlap(edge_offset))>0:
                new_nodes.append(overlap[0].point_at(0.5))
            # 第三种情况：不相交，包括直线平行/直线和圆
            else:
                new_nodes.append(pre_edge_offset.e)
                new_nodes.append(edge_offset.s)                
            pre_edge_offset=edge_offset
        new_loop=Loop.from_nodes(new_nodes)
        if not split:
            return [new_loop]
    def has_self_intersection(self) -> bool:  # ok, str ok
        """判断是否有自相交"""
        # self.edges们两两判断是否相交/重叠
        # 判断的时候每条线段的有效范围是[0,1)->[s,e)；只算头，不算尾巴
        for i,ei in enumerate(self.edges):
            if self.prepared is not None:
                neighbors=self.prepared.query(ei.get_mbb(),tol=Const.TOL_DIST)
            else: neighbors=self.edges[i:]
            for ej in neighbors:
                if ei is ej: continue
                # 相交
                intersection=ei.intersection(ej)
                for p in intersection:
                    if not p.equals(ei.e) and not p.equals(ej.e): return True
                # 重叠
                if ei.s.is_on_edge(ej) and not ei.s.equals(ej.e): return True
                if ej.s.is_on_edge(ei) and not ej.s.equals(ei.e): return True
        return False
    def covers(self,other:Geom,count_mode:str="or")->bool:  # ok
        """环覆盖其他对象"""
        if not (count_mode=="or" or count_mode=="xor"): 
            raise ValueError("Invalid count mode.")
        if GeomRelation.Outside in Geom.mbb_relation(self.get_mbb(),other.get_mbb()):  # 包围盒不满足则不满足
            return False
        if isinstance(other,Node):
            return self._covers_node(other,count_mode)
        if isinstance(other,Edge):
            return self._covers_edge(other,count_mode)
        if isinstance(other,Polyedge):
            return self._covers_polyedge(other,count_mode)
        if isinstance(other,Polygon):
            return self._covers_polygon(other,count_mode)
        return False
    def contains(self,other:Geom,count_mode:str="or")->bool:  # ok
        """环包含其他对象"""
        if not (count_mode=="or" or count_mode=="xor"): raise ValueError("Invalid count mode.")
        if isinstance(other,Node):
            return self._contains_node(other,count_mode)
        if isinstance(other,Edge):
            return self._contains_edge(other,count_mode)
        if isinstance(other,Polyedge):
            return self._contains_polyedge(other,count_mode)
        if isinstance(other,Polygon):
            return self._contains_polygon(other,count_mode)
        return False    
    def _relation_with_node(self,other:Node,count_mode:str="or")->GeomRelation:
        """判断点和环的关系"""
        rel=Geom.mbb_relation(self.get_mbb(),other.get_mbb())
        if rel==[GeomRelation.Outside]:  # 包围盒不满足则不满足
            return rel[0]
        # 先判断是否在边界上
        if self.prepared is not None:
            neighbor_edges=self.prepared.query(other.get_mbb(),tol=Const.TOL_DIST)
        else: neighbor_edges=self.edges
        for edge in neighbor_edges:
            if edge.touches_node(other):
                return GeomRelation.OnBoundary
        # 判断内外：射线法，计算环“真实”穿越射线的次数
        # 向上：+1；向下：-1；向上+终点 或 向下+起点：不计
        # 对于"xor"模式：偶数在外，奇数在内；对于"or"模式，0在外，非0在内
        max_x=self.get_mbb()[1].x
        ray=LineSeg(other,Node(max_x+Const.TOL_DIST*2,other.y))
        if self.prepared is not None:
            neighbor_edges=self.prepared.query(ray.get_mbb(),tol=Const.TOL_DIST)
        else: neighbor_edges=self.edges
        cross_count=0
        for edge in neighbor_edges:
            int_pts=ray.intersection(edge)
            if len(int_pts)==0:continue
            if len(int_pts)==2 and int_pts[0].equals(int_pts[1]):int_pts=[int_pts[1]]
            for p in int_pts:
                is_up,is_down=False,False
                t=edge.get_param(p)
                tangent=edge.tangent_at(t)
                if tangent.equals(Vec3d.X) or tangent.equals(-Vec3d.X):  # 相切
                    if t!=0 and t!=1: continue  # 只是经过一下就不计
                    normal=edge.principal_normal_at(t)
                    if normal.y>0 and t==0 or normal.y<0 and t==1: is_up=True  # 切点->一二象限 或 三四象限->切点
                    if normal.y<0 and t==0 or normal.y>0 and t==1: is_down=True  # 切点->三四象限 或 一二象限->切点
                elif tangent.y>0: is_up=True
                elif tangent.y<0: is_down=True
                if is_up and t!=1: cross_count+=1
                if is_down and t!=0: cross_count-=1
        if count_mode=="or": return GeomRelation.Inside if cross_count!=0 else GeomRelation.Outside
        if count_mode=="xor": return GeomRelation.Inside if cross_count%2==1 else GeomRelation.Outside
    def _touches_node(self,other:Node)->bool:  # ok
        """点在环的边界上"""
        return self._relation_with_node(other) is GeomRelation.OnBoundary
    def _contains_node(self,other:Node,count_mode:str="or")->bool:  # ok
        """环包含点"""
        return self._relation_with_node(other,count_mode) is GeomRelation.Inside
    def _covers_node(self,other:Node,count_mode:str="or")->bool:  # ok
        """环覆盖点"""
        return self._relation_with_node(other,count_mode) is not GeomRelation.Outside
    def clips_edge(self,other:Edge,keep:list[GeomRelation]=None,count_mode:str="or")->list[Edge]:
        """用环剪切边.

        Args:
            other (Edge): 被剪切的边.
            keep (list[GeomRelation], optional): 保留哪些. Defaults to [GeomRelation.Inside].
            count_mode (str="or"|"xor", optional): 环内外的判断规则. Defaults to "or".

        Returns:
            list[Edge]: 保留的部分.
        """
        keep=keep or [GeomRelation.Inside]
        segs=self._cuts_edge(other,count_mode=count_mode)
        return sum([segs[rel] for rel in keep],[])
    def _cuts_edge(self,other:Edge,count_mode:str="or")->dict[GeomRelation,list[Edge]]:
        segs={GeomRelation.Inside:[],
              GeomRelation.Outside:[],
              GeomRelation.OnBoundary:[],
              GeomRelation.Intersect:[]}
        break_points=[]
        if self.prepared is not None:
            neighbor_edges=self.prepared.query(other.get_mbb(),tol=Const.TOL_DIST)
        else: neighbor_edges=self.edges
        for edge in neighbor_edges:
            if edge.s.is_on_edge(other): break_points.append(edge.s)
            elif edge.e.is_on_edge(other): break_points.append(edge.e)
            else: break_points+=other.intersection(edge)
        if len(break_points)>0: 
            segs[GeomRelation.Intersect].append(other)
        break_points+=[other.s,other.e]
        break_points.sort(key=lambda p:other.get_param(p))
        s=break_points[0]
        for e in break_points:
            if e.equals(s): continue
            seg=other.slice_between(s,e)
            mid=seg.point_at(0.5)
            rel=self._relation_with_node(mid,count_mode)
            if rel is not GeomRelation.OnBoundary: segs[rel].append(seg)
            else:  # 中点在环上时补充判断起终点，防止出现中点在误差范围内而端点不在的情况
                rel_s=self._relation_with_node(s,count_mode)
                if rel_s is not GeomRelation.OnBoundary: segs[rel_s].append(seg)
                else:
                    rel_e=self._relation_with_node(e,count_mode)
                    if rel_e is not GeomRelation.OnBoundary: segs[rel_e].append(seg)
                    else: segs[rel].append(seg)
            s=e
        return segs
    def _relation_with_edge(self,other:Edge,count_mode:str="or")->set[GeomRelation]:
        res=set()
        segs=self._cuts_edge(other,count_mode=count_mode)
        res=set([rel for rel in segs if len(segs[rel])>0])
        return res
    def _relation_with_loop(self,other:"Loop",count_mode:str="or")->set[GeomRelation]:
        res=set()
        for edge in other.edges:
            rel=self._relation_with_edge(edge,count_mode)
            res|=rel
            if len(res)==4: break
        return res
    def _covers_edge(self,other:Edge,count_mode:str="or")->bool:  # ok
        """环覆盖边"""
        rel=self._relation_with_edge(other,count_mode)
        return GeomRelation.Outside not in rel
    def _contains_edge(self,other:Edge,count_mode:str="or")->bool:  # ok
        """环包含边"""
        rel=self._relation_with_edge(other,count_mode)
        return GeomRelation.Inside in rel and len(rel)==1
    def _covers_polyedge(self,other:Polyedge,count_mode:str="or")->bool:  # ok
        """环覆盖多段线/环"""
        for edge in other.edges:
            if not self._covers_edge(edge,count_mode):
                return False
        return True
    def _contains_polyedge(self,other:Polyedge,count_mode:str="or")->bool:  # ok
        """环包含多段线/环"""
        for edge in other.edges:
            if not self._contains_edge(edge,count_mode):
                return False
        return True
    def _covers_polygon(self,other:"Polygon",count_mode:str="or")->bool:  # ok
        return self._covers_polyedge(other.shell,count_mode)
    def _contains_polygon(self,other:"Polygon",count_mode:str="or")->bool:  # ok
        return self._contains_polyedge(other.shell,count_mode)    
    def fillet(self,radius:float,mode:str="amap",quad_segs:int=16)->"Loop": 
        """倒圆角.

        Parameters
        ----------
        radius : float
        mode : str, optional, by default "amap"
            "amap": 保证radius，保证loop合法，倒角尽可能多的顶点;
            "raw": 保证radius，倒角所有顶点，不保证loop合法（可能自相交）;
            "relax": 倒角所有顶点，保证loop合法，不保证radius;
        quad_segs : int, optional, by default 16
        """
        new_nodes=[]
        if mode=="raw": 
            for i in range(len(self.edges)):
                pre_edge=self.edges[i-1]
                edge=self.edges[i]
                arc=pre_edge.fillet_with(edge,radius)
                new_nodes+=arc.fit().nodes
        if mode=="relax": 
            relaxed_radius=[radius]*len(self.edges)
            for i in range(len(self.edges)):
                pre_edge=self.edges[i-1]
                edge=self.edges[i]
                next_edge=self.edges[(i+1)%len(self.edges)]
                pre_arc=pre_edge.fillet_with(edge,relaxed_radius[i])
                next_arc=edge.fillet_with(next_edge,relaxed_radius[(i+1)%len(self.edges)])
                d1=edge.s.dist(pre_arc.e)
                d2=edge.e.dist(next_arc.s)
                if d1+d2>edge.length:
                    pass
                arc=pre_edge.fillet_with(edge,radius)
                new_nodes+=arc.fit().nodes
        return Loop.from_nodes(new_nodes)

class Polygon(Geom): 
    """多边形"""
    def __init__(self,shell:Loop,holes:list[Loop]=None,make_valid:bool=True,prepare:bool=False) -> None:
        super().__init__()
        holes=holes or []
        if not (isinstance(shell,Loop) and all([isinstance(hole,Loop) for hole in holes])): raise TypeError()
        self.shell=shell
        self.holes=holes[:]
        if make_valid: 
            self._make_valid()
        else: self.is_valid=False
        if self.is_valid and prepare: self.prepare()
    def _make_valid(self)->None:  # ok
        from lib.geom_algo import BooleanOperation
        # 1.修正内外环方向
        if self.shell.area<0: self.shell=self.shell.reversed()
        for i,hole in enumerate(self.holes): 
            if hole.area>0: self.holes[i]=hole.reversed()
        # 2.判断polygon有效性iff重建拓扑关系之后的正环数量==1
        cond=lambda depth:depth==1  # Loop union
        valid_polygon=BooleanOperation._rebuild_loop_topology(self.all_loops,condition=cond)
        if len(valid_polygon)==1:
            self.shell=valid_polygon[0].shell
            self.holes=valid_polygon[0].holes
            self.is_valid=True
        else:
            self.is_valid=False
    def prepare(self)->None:
        self.shell.prepare()
        for hole in self.holes:
            hole.prepare()
    def __copy__(self)->Self:
        return Polygon(self.shell,self.holes,make_valid=False)
    def __deepcopy__(self)->Self:
        return Polygon(deepcopy(self.shell),deepcopy(self.holes),make_valid=False)
    def get_mbb(self) -> tuple[Node, Node]:
        return self.shell.get_mbb()
    def to_array(self)->tuple[np.ndarray,np.ndarray]:
        return self.shell.to_array(),[hole.to_array() for hole in self.holes]
    @property
    def all_loops(self)->list[Loop]:  # ok
        return [self.shell]+self.holes
    @property
    def area(self)->float:  # ok
        if hasattr(self,"_area"): return self._area
        self._area=sum([ring.area for ring in self.all_loops])
        return self._area
    @property
    def edges(self)->Generator[Edge,None,None]:
        for loop in self.all_loops: yield from loop.edges
    @property
    def nodes(self)->Generator[Node,None,None]:
        for loop in self.all_loops: yield from loop.nodes
    def is_identical_to(self,other:"Polygon")->bool:
        if not (len(self.holes)==len(other.holes) and
                self.shell.is_identical(other.shell)):
            return False
        for i in self.holes:
            for j in other.holes:
                if i.is_identical(j): break
            else: return False
        return True
    def covers(self,other:Geom)->bool: 
        """多边形覆盖其他对象. 

        **Attention:**
            Polygon包含Loop的意思是包含边而非区域，要判断区域需构造Polygon
        """
        rel=Geom.mbb_relation(self.get_mbb(),other.get_mbb())
        if GeomRelation.Outside in rel: return False  # 包围盒不满足则不满足
        if isinstance(other,Node):
            if not self.shell.covers(other): return False
            for hole in self.holes:
                if hole.contains(other):
                    return False
            return True
        if isinstance(other,Edge):
            if not self.shell.covers(other): return False
            for hole in self.holes:
                if GeomRelation.Inside in hole._relation_with_edge(other):
                    return False
            return True
        if isinstance(other,Polyedge):
            for edge in other.edges:
                if not self.covers(edge): 
                    return False
            return True
        if isinstance(other,Polygon):
            from lib.geom_algo import BooleanOperation
            u=BooleanOperation.union([self,other])
            return len(u)==1 and self.is_identical_to(u[0])

    def contains(self,other:Geom)->bool: 
        """多边形包含其他对象. 注意：Polygon包含Loop的意思是包含边而非区域，要判断区域需构造Polygon"""
        rel=Geom.mbb_relation(self.get_mbb(),other.get_mbb())
        if GeomRelation.Outside in rel or GeomRelation.OnBoundary in rel: return False  # 包围盒不满足则不满足
        if isinstance(other,Node):
            if not self.shell.contains(other): return False
            for hole in self.holes:
                if hole.covers(other):
                    return False
            return True
        if isinstance(other,Edge):
            if not self.shell.contains(other): return False
            for hole in self.holes:
                rel=hole._relation_with_edge(other)
                if GeomRelation.Inside in rel or GeomRelation.OnBoundary in rel:
                    return False
            return True
        if isinstance(other,Polyedge):
            for edge in other.edges:
                if not self.contains(edge): 
                    return False
            return True
        if isinstance(other,Polygon):
            if not self.covers(other):return False
            for loop1 in self.all_loops:
                for loop2 in other.all_loops:
                    if GeomRelation.Intersect in loop1._relation_with_loop(loop2):
                        return False        
            return True
    def offset(self,dist:float)->list["Polygon"]:
        shells=self.shell.offset(dist)
        holes=[]
        for hole in self.holes:
            holes+=hole.offset(dist)
        
        return ...
    def closest_point(self,other:Geom)->Node:
        if isinstance(other,Node):
            min_dist=Const.MAX_VAL
            res=None
            for loop in self.all_loops:
                for edge in loop.edges:
                    p=edge.closest_point(other)
                    if (d:=p.dist(other))<min_dist:
                        res,min_dist=p,d
            return res
    def clips_edge(self,other:Edge,keep:list[GeomRelation]=None)->list[Edge]:
        """用多边形剪切边.

        Args:
            other (Edge): 被剪切的边.
            keep (list[GeomRelation], optional): 保留哪些. Defaults to [GeomRelation.Inside].

        Returns:
            list[Edge]: 保留的部分.
        """
        keep=keep or [GeomRelation.Inside]

        shell_cuts=self.shell._cuts_edge(other)
        res={GeomRelation.Inside:[],
             GeomRelation.Outside:shell_cuts[GeomRelation.Outside],
             GeomRelation.OnBoundary:shell_cuts[GeomRelation.OnBoundary],
        }
        in_between=shell_cuts[GeomRelation.Inside]
        for hole in self.holes:
            outside_hole=[]
            for seg in in_between:
                hole_cuts=hole._cuts_edge(seg)
                res[GeomRelation.Outside]+=hole_cuts[GeomRelation.Inside]
                res[GeomRelation.OnBoundary]+=hole_cuts[GeomRelation.OnBoundary]
                outside_hole+=hole_cuts[GeomRelation.Outside]
            in_between=outside_hole
        res[GeomRelation.Inside]=in_between
        return sum([res[rel] for rel in keep],[])

class GeomUtil:
    """几何工具类"""
    @staticmethod
    def find_or_insert_node(target_node:Node,nodes:list[Node],copy=False)->Node:
        for node in nodes:
            if target_node.equals(node):
                return node if not copy else Node(node.x,node.y,node.z)
        nodes.append(target_node)
        return target_node
    @staticmethod
    def add_edge_to_node_in_order(node:Node, edge:Edge)->None: 
        """有序插入：第一关键字角度，第二关键字曲率半径（左正右负）"""
        ang=edge.tangent_at(0).angle
        radius=edge.radius_at(0,signed=True)
        comp=Const.cmp_ang
        i=0
        while i<len(node.edge_out):
            angle_i=node.edge_out[i].tangent_at(0).angle
            radius_i=node.edge_out[i].radius_at(0,signed=True)
            comp_angle=comp(angle_i,ang)
            if comp_angle>0: break  # 先按角度tol升序
            elif comp_angle==0:  # 角度tol相等时
                comp_radius=Edge.compare_curvature_by_radius(radius_i,radius)
                if comp_radius>0: break  # 按曲率tol升序
                elif comp_radius==0:  # 曲率tol也相等时
                    if angle_i>ang: break  # 按角度严格升序
                    elif angle_i==ang:  # 角度也严格相等时
                        if radius_i>radius_i: break  # 按曲率严格升序
            i+=1
        node.edge_out.insert(i,edge)
    @staticmethod
    def find_next_edge_out(node:Node, pre_edge:Edge)->Edge:
        """按角度和半径找下一条出边，确保内环优先逆时针方向"""
        op=pre_edge.opposite()
        pre_angle=op.tangent_at(0).angle  # 入边的角度
        pre_radius=op.radius_at(0,signed=True)  # 入边的半径        
        i=len(node.edge_out)-1
        comp=Const.cmp_ang
        while i>=0:
            angle_i=node.edge_out[i].tangent_at(0).angle
            radius_i=node.edge_out[i].radius_at(0,signed=True)
            if comp(angle_i,pre_angle)<0: 
                break  # 取角度比前一条出边小的第一条边
            if comp(angle_i,pre_angle)==0:
                if Edge.compare_curvature_by_radius(radius_i,pre_radius)<0:
                    break  # 角度相同的，比较带符号曲率，取曲率比前一条出边小的第一条边
            i-=1
        return node.edge_out[i]
    

