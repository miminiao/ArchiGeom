import math
from lib.utils import Constant
import numpy as np
class Tensor:
    dim:tuple[int,...]
    @property
    def const(self)->Constant:
        return Constant.get()
class Vector(Tensor):...
class Vec3d(Vector):
    dim=(3,1)
    def __init__(self,x:float,y:float,z:float=0.0) -> None:
        self.x,self.y,self.z=x,y,z
    @classmethod
    def X(cls)->"Vec3d": return cls(1,0,0)
    @classmethod
    def Y(cls)->"Vec3d": return cls(0,1,0)
    @classmethod
    def Z(cls)->"Vec3d": return cls(0,0,1)
    def __getitem__(self,index:int)->float:
        if index>=3 or index<-3: raise IndexError
        match index%3:
            case 0: return self.x
            case 1: return self.y
            case 2: return self.z
    def __repr__(self) -> str:
        return f"Vec3d({self.x},{self.y},{self.z})"
    def __add__(self,other:"Vec3d")->"Vec3d":
        return Vec3d(self.x+other.x,self.y+other.y,self.z+other.z)
    def __sub__(self,other:"Vec3d")->"Vec3d":
        return Vec3d(self.x-other.x,self.y-other.y,self.z-other.z)
    def __neg__(self)->"Vec3d":
        return Vec3d(-self.x,-self.y,-self.z)
    def __mul__(self,scaler:float)->"Vec3d":
        return Vec3d(self.x*scaler,self.y*scaler,self.z*scaler)
    def __truediv__(self,divider:float)->"Vec3d":
        return Vec3d(self.x/divider,self.y/divider,self.z/divider)
    def equals(self,other:"Vec3d")->bool:
        return (self-other).length<self.const.TOL_DIST
    def dot(self,other:"Vec3d")->float:
        return self.x*other.x+self.y*other.y+self.z*other.z
    def cross(self,other:"Vec3d")->"Vec3d":
        return Vec3d(self.y*other.z-self.z*other.y,self.z*other.x-self.x*other.z,self.x*other.y-self.y*other.x)
    @property
    def length(self)->float:
        return (self.x**2+self.y**2+self.z**2)**0.5
    def is_zero(self,is_unit=False)->bool:
        if is_unit: return self.length<self.const.TOL_VAL
        else: return self.length<self.const.TOL_DIST
    @property
    def angle(self)->float:
        """角度范围[0,2pi), 含误差"""
        if self.length<self.const.TOL_VAL: return 0
        cosX=self.x/self.length
        cosY=self.y/self.length 
        angle=math.acos(cosX) if cosY>=0 else 2*math.pi-math.acos(cosX)
        if 2*math.pi-angle<self.const.TOL_ANG: angle-=2*math.pi
        return angle
    def unit(self)->"Vec3d":
        return self/self.length
    def angle_between(self,other:"Vec3d")->float:
        """不分正负"""
        return math.acos(self.dot(other)/self.length/other.length)
    def angle_to(self,other:"Vec3d")->float:
        """求到other的旋转角[0,2pi)"""
        res=other.angle-self.angle
        if res<0: res+=2*math.pi
        return res
    def rotate2d(self,angle:float)->"Vec3d":
        return Vec3d(math.cos(angle)*self.x-math.sin(angle)*self.y,math.sin(angle)*self.x+math.cos(angle)*self.y,self.z)
    def to_array(self)->np.ndarray:
        return np.array([self.x,self.y,self.z]).T
    def to_list(self)->list[float]:
        return [self.x,self.y,self.z]
    def to_vec4d(self,w:float=0)->"Vec4d":
        return Vec4d(self.x,self.y,self.z,w)
class Vec4d(Vector):
    dim=(4,1)
    def __init__(self,x:float,y:float,z:float,w:float=1) -> None:
        self.x,self.y,self.z,self.w=x,y,z,w
    def __repr__(self) -> str:
        return f"Vec4d({self.x},{self.y},{self.z},{self.w})"        
    @classmethod
    def X(cls)->"Vec4d": return cls(1,0,0,0)
    @classmethod
    def Y(cls)->"Vec4d": return cls(0,1,0,0)
    @classmethod
    def Z(cls)->"Vec4d": return cls(0,0,1,0)        
    @classmethod
    def W(cls)->"Vec4d": return cls(0,0,0,1)            
    def to_list(self)->list[float]:
        return [self.x,self.y,self.z,self.w]
    def to_array(self)->np.ndarray:
        return np.array([self.x,self.y,self.z,self.w]).T    
class Matrix(Tensor):
    def __init__(self,mat:list[list[float]]) -> None:
        assert len(mat)==self.dim[0] and all([len(row)==self.dim[1] for row in mat])
        self.mat=np.array(mat)
    def __getitem__(self,index:tuple[int,int]|int)->float|list[float]:
        if isinstance(index,int): 
            if index>=self.dim[1] or index<-self.dim[1]: raise IndexError
            return list(self.mat[index])
        elif isinstance(index,tuple):
            if index[0]>=self.dim[0] or index[0]<-self.dim[0] or index[1]>=self.dim[1] or index[1]<-self.dim[1]: raise IndexError
            return self.mat[index]
    def to_array(self)->np.ndarray:
        return np.array(self.mat)
class Mat3d(Matrix):
    dim=(3,3)
    def __init__(self,mat:list[list[float]]) -> None:
        super().__init__(mat)    
    @classmethod
    def from_column_vecs(cls,columns:list[Vec3d])->"Mat3d":
        return cls([[columns[j][i] for j in range(cls.dim[1])] for i in range(cls.dim[0])])
    @classmethod
    def from_row_vecs(cls,rows:list[Vec3d])->"Mat3d":
        return cls([vec.to_list() for vec in rows])
    def determinant(self)->float:
        return np.linalg.det(self.mat)
    def inverse(self)->"Mat3d":
        return Mat3d(list(np.linalg.inv(self.mat)))
    def transpose(self)->"Mat3d":
        return Mat3d(list(self.mat.T))
    def __matmul__(self,other):
        if isinstance(other,Vec3d):
            return Vec3d(*(self.mat@other.to_array()))
        elif isinstance(other,Mat3d):
            return Mat3d(list(self.mat@other.mat))
    def __mul__(self,other:int|float):
        return Mat3d(list(self.mat*other))
class Mat4d(Matrix):
    """4x4矩阵"""
    dim=(4,4)
    def __init__(self,mat:list[list[float]]) -> None:
        super().__init__(mat)
    @classmethod
    def from_row_vecs(cls,rows:list[Vec4d])->"Mat4d":
        return cls([vec.to_list() for vec in rows])
    @classmethod
    def from_column_vecs(cls,columns:list[Vec4d])->"Mat4d":
        return cls([[columns[j][i] for j in range(cls.dim[1])] for i in range(cls.dim[0])])    
    def __matmul__(self,other):
        if isinstance(other,Vec4d):
            return Vec4d(*(self.mat@other.to_array()))
        elif isinstance(other,Mat4d):
            return Mat4d(list(self.mat@other.mat))
    def __mul__(self,other:int|float):
        return Mat4d(list(self.mat*other))    
    def inverse(self)->"Mat4d":
        return Mat4d(list(np.linalg.inv(self.mat)))
if __name__=="__main__":
    a=[[0,1,2],[3,4,5],[6,7,8]]
    m=Mat3d(a)
    b=[1,2,3]
    v=Vec3d(*b)
    print(m@v,(m*2).mat)