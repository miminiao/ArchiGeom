"""通用工具类"""

import json
import pathlib
import math
from copy import copy
from time import time,sleep
from typing import Any,Callable,Self,Protocol
from functools import wraps

# -----------------------------------------------------------------------------

class Timer:
    """计时器。

    Args:
        tag (str): 测试标签。

    Examples:
        - 作为函数装饰器: 

        >>> Timer.enable()
        >>> @Timer
        >>> def foo(): ...
        >>> foo()
        foo 0.0

        - 作为上下文管理器: 

        >>> Timer.enable()
        >>> with Timer(tag="bar"): ...
        bar 0.0

    """
    _enabled=False
    logs=[]
    def __init__(self,func=None,tag:str="") -> None:
        self._func=func
        self._tag=tag
        self._instance=None
    def __enter__(self):
        self._start=time()
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self._enabled:
            disc=self._tag
            t=time()-self._start
            self.logs.append((disc,t))
            print(disc,t)
    def __get__(self,instance,owner):
        self._instance=instance
        return self
    def __call__(self,*args,**kwargs):
        t0=time()
        ret=self._func(self._instance,*args,**kwargs)
        if self._enabled:
            disc=self._tag+self._func.__name__
            t=time()-t0
            self.logs.append((disc,t))
            print(disc,t)
        return ret
    @classmethod
    def enable(cls): cls._enabled=True
    @classmethod
    def disable(cls): cls._enabled=False

# -----------------------------------------------------------------------------

class ConstVarDescriptor:
    def __init__(self,attr_name:str):
        self.attr_name=attr_name
    def __get__(self,instance,owner):
        return getattr(owner._stack[-1],self.attr_name)

class Constant:
    """全局常量类。

    Args:
        MAX_VAL (float, optional): 正无穷.
        TOL_VAL (float, optional): 浮点容差.
        TOL_DIST (float, optional): 距离容差.
        TOL_AREA (float, optional): 面积容差.
        TOL_ANG (float, optional): 角度容差/rad.

    Examples:
        - 直接使用类属性访问当前常量

        >>> print(Constant.TOL_DIST)
        1e-2

        - 使用上下文管理器创建自定义的常量

        >>> with Constant(tol_dist=1e-1):
        ...     print(Constant.TOL_DIST)
        1e-1

        - 使用比较函数

        >>> cmp=Constant.cmp_dist
        >>> print(cmp(1.000,1.001))
        0
    """

    _arg_names=("MAX_VAL","TOL_VAL","TOL_DIST","TOL_AREA","TOL_ANG")
    _stack:list[Self]=[]

    MAX_VAL=ConstVarDescriptor('_MAX_VAL')
    """正无穷（当前全局生效值）"""
    TOL_ANG=ConstVarDescriptor('_TOL_ANG')
    """角度容差/rad（当前全局生效值）"""
    TOL_AREA=ConstVarDescriptor('_TOL_AREA')
    """面积容差（当前全局生效值）"""
    TOL_DIST=ConstVarDescriptor('_TOL_DIST')
    """距离容差（当前全局生效值）"""
    TOL_VAL=ConstVarDescriptor('_TOL_VAL')
    """浮点容差（当前全局生效值）"""

    DEFAULT:Self=None
    """默认值，从配置文件env.json['CONSTANTS']读取"""

    def __init__(self,*,
                 MAX_VAL:float=None,
                 TOL_VAL:float=None,
                 TOL_DIST:float=None,
                 TOL_AREA:float=None,
                 TOL_ANG:float=None,
                 ) -> None:
        self._MAX_VAL=MAX_VAL or self.DEFAULT._MAX_VAL
        self._TOL_VAL=TOL_VAL or self.DEFAULT._TOL_VAL
        self._TOL_DIST=TOL_DIST or self.DEFAULT._TOL_DIST
        self._TOL_AREA=TOL_AREA or self.DEFAULT._TOL_AREA
        self._TOL_ANG=TOL_ANG or self.DEFAULT._TOL_ANG
    @classmethod
    def cmp_val(cls,x:float,y:float)->int:
        """浮点数的比较函数.
        
        Returns:
            (>, <, ==) -> (1, -1, 0)
        """
        if abs(x-y)<cls.TOL_VAL: return 0
        elif x>y: return 1
        else: return -1
    @classmethod        
    def cmp_dist(cls,x:float,y:float)->int:
        """距离的比较函数.
        
        Returns:
            (>, <, ==) -> (1, -1, 0)
        """
        if abs(x-y)<cls.TOL_DIST: return 0
        elif x>y: return 1
        else: return -1
    @classmethod
    def cmp_area(cls,x:float,y:float)->int:
        """面积的比较函数.
        
        Returns:
            (>, <, ==) -> (1, -1, 0)
        """        
        if abs(x-y)<cls.TOL_AREA: return 0
        elif x>y: return 1
        else: return -1
    @classmethod
    def cmp_ang(cls,x:float,y:float,periodic:bool=True)->int:
        """角度的比较函数.
        
        Args:
            x,y (float): 角度/rad
            periodic (bool,optional): True时直接比较；否则先将x和y换算到[0,2pi)范围
        Returns:
            (>, <, ==) -> (1, -1, 0)
        """
        if not periodic:
            x%=math.pi*2
            y%=math.pi*2
            if math.pi*2-abs(x-y)<cls.TOL_ANG: return 0         
        if abs(x-y)<cls.TOL_ANG: return 0
        elif x>y: return 1
        else: return -1
    def __enter__(self):
        Constant._stack.append(copy(self))
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        Constant._stack.pop()
    @classmethod
    def _init_default(cls)-> Self:
        with open(pathlib.Path(__file__).parent.parent/"env.json",'r') as f:
            js=json.load(f)["CONSTANTS"]
        args={name:float(js[name]) for name in cls._arg_names}
        cls.DEFAULT=cls(**args)
        cls._stack.append(cls.DEFAULT)

Constant._init_default()

# -----------------------------------------------------------------------------

class ListTool:
    @staticmethod
    def distinct(a:list[float], /,*,
                 tol:float=0,
                 cmp_func:Callable[[float,float],int]=None
    ) -> list[float]:
        """排序并去除list[float]中的重复元素.

        Args:
            tol (float, optional): 精度容差. 当cmp_func不为None时失效. Defaults to 0.
            cmp_func (Callable[[float,float],int], optional): 比较函数. 不为None时覆盖tol. Defaults to None.
        """
        if len(a)==0: return
        if cmp_func is None:
            tol=tol or Constant.TOL_VAL
            cmp_func=lambda x,y:0 if abs(x-y)<=tol else 1
        tmp=sorted(a)
        res=[tmp[0]]
        for x in tmp:
            if cmp_func(x,res[-1])!=0:
                res.append(x)
        return res
    
    @staticmethod
    def bsearch[T](a:list[T], x:T, /, *, 
                   cmp_func:Callable[[T,T],int]=None,
    ) -> tuple[bool, int]:
        """在(sorted)list中二分查找x的index，找不到则返回应当插入的位置.

        Args:
            cmp_func (Callable[[T,T],int], optional): 比较函数, (>,<,==) -> (1,-1,0). Defaults to None.

        Returns:
            bool: 是否查找成功.
            int: x的index(找到了) / 应该插入到的index(没找到).
        """
        cmp_func=cmp_func or (lambda a,b: a-b)
        l,r=0,len(a)-1
        while l<=r:
            m=(l+r)//2
            match cmp_func(x,a[m]):
                case t if t<0: r=m-1
                case t if t>0: l=m+1
                case 0: return True, m
        return False, l
    @staticmethod
    def first[T](a:list[T],/,*,cond:Callable[[T],bool])->int:
        """在list中查找第0个满足cond条件的对象的index"""
        for i,item in enumerate(a):
            if cond(item): return i
        return -1
    @staticmethod
    def get_nth[T](a:list[T],nth:int,/,*,key:Callable[[T],float]=None)->int:
        """在list中查找第n小的元素的index，n starts from 0"""
        idx=list(range(len(a)))
        key=key or (lambda x:x)
        return ListTool._get_nth(a[:],idx,nth,key=key,l=0,r=len(a)-1)
    @staticmethod
    def _get_nth[T](a:list[T],idx:list[int],nth:int,l:int,r:int,key:Callable[[T],float])->int:
        # 参照快速排序，但每次只需要递归包含第n位的一侧, 均摊时间O(n)
        if l==r: return idx[l]
        i,j,mid=l,r,key(a[(l+r)//2])
        while True:
            while i<r and key(a[i])<mid: i+=1
            while l<j and key(a[j])>mid: j-=1
            if i<=j:
                a[i],a[j]=a[j],a[i]
                idx[i],idx[j]=idx[j],idx[i]
                i,j=i+1,j-1
            else: break
        if j+1==nth==i-1: return idx[nth]  # 第n位已确定在中间
        elif nth<=j and l<=j: return ListTool._get_nth(a,idx,nth,l,j,key=key)  # 第n位在左边
        elif nth>=i and i<=r: return ListTool._get_nth(a,idx,nth,i,r,key=key)  # 第n位在右边
    @staticmethod
    def qsort[T](a:list[T],/,*,key:Callable[[T],float]=None):
        """排序"""
        key=key or (lambda x:x)
        ListTool._qsort(a,0,len(a)-1,key=key)
    @staticmethod
    def _qsort[T](a:list[T],l:int,r:int,key:Callable[[T],float]):
        i,j,mid=l,r,key(a[(l+r)//2])
        while True:
            while i<r and key(a[i])<mid: i+=1
            while l<j and key(a[j])>mid: j-=1
            if i<=j:
                a[i],a[j]=a[j],a[i]
                i,j=i+1,j-1
            else: break
        if l<j: ListTool._qsort(a,l,j,key=key)
        if i<r: ListTool._qsort(a,i,r,key=key)

class StopRetry(Exception):
    """Exception用于提前中止retry.
    Args:
        cause (Exception): 中止retry的原因.
    """
    def __init__(self, cause:Exception, *args) -> None:
        super().__init__(cause,*args)
        self.cause=cause

def retry(max_times:int=10,delay:float=0):
    """Decorator: 失败后重试, 超过上限抛出错误.

    Args:
        max_times (int, optional): 最大次数. Defaults to 10.
        interval (float, optional): 间隔等待时间/秒. Defaults to 0.

    Examples:
        重试执行3次
        >>> @retry(max_times=3)
        >>> def foo(): 
        ...     raise RuntimeError("bar")
        RuntimeError: bar. Retrying...
        RuntimeError: bar. Retrying...
        RuntimeError: bar. Retrying...
        RuntimeError: bar

        提前中止重试
        >>> @retry(max_times=3)
        >>> def foo(): 
        ...     raise StopRetry(StopRuntimeError("bar"))
        RuntimeError: bar
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            retry_count=0
            while retry_count<max_times:
                try:
                    return func(*args,**kwargs)
                except StopRetry as e:
                    raise e.cause
                except Exception as e:
                    print(f"{e.__class__.__name__}: {e}. Retrying...")
                    last_err=e
                    retry_count+=1
                    if delay>0: sleep(delay)
            raise last_err
        return wrapper
    return decorator

class SupportsCompareFunc(Protocol):
    """支持cls._cmp()比较函数"""
    _cmp:Callable[[float,float],int]

class SupportsCompareOps(Protocol):
    """支持>,>=,<,<=比较"""
    def __lt__(self,other)->bool: ...
    def __gt__(self,other)->bool: ...
    def __le__(self,other)->bool: ...
    def __ge__(self,other)->bool: ...

class ComparerInjector[T]:
    """比较函数注入器，给指定的类注入比较方法，
    使其支持SupportsCompareFunc Protocol，及SupportsCompareOps Protocol（可选）.

    Args:
        cls (type[T]): 需要注入比较函数的类.
        func (Callable[[T,T],int]): 注入的比较函数.
        override_ops (bool, optional): 是否重载比较运算符以支持SupportsCompareOps Protocol. Defaults to False.

    Examples:
        作为上下文管理器
        >>> compare_wall=lambda x,y:Constant.cmp_dist(x.width,y.width)
        >>> with ComparerInjector(Wall,compare_wall,override_ops=True):
        ...     print(Wall(width=100)>Wall(width=50))
        ...     print(Wall._cmp(Wall(width=50),Wall(width=50.001)))
        True
        0
    """
    def __init__(self,cls:type[T],func:Callable[[T,T],int],override_ops:bool=False) -> None:
        self._cls=cls
        self._func=func or (lambda x,y:0)
        self._override_ops=override_ops

    def __enter__(self):
        if "_cmp" in self._cls.__dict__:
            self._previous_func=self._cls._cmp
        else: 
            self._previous_func=None
        self._inject(self._func)
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self._previous_func is not None:
            self._inject(self._previous_func)
        else: 
            self._eject()
    def _inject(self,func)->None:
        self._cls._cmp=func
        if self._override_ops:
            self._cls.__lt__=lambda self,other: func(self,other)<0
            self._cls.__gt__=lambda self,other: func(self,other)>0
            self._cls.__le__=lambda self,other: func(self,other)<=0
            self._cls.__ge__=lambda self,other: func(self,other)>=0
    def _eject(self)->None:
        delattr(self._cls,"_cmp")
        if self._override_ops:
            delattr(self._cls,"__lt__")
            delattr(self._cls,"__gt__")
            delattr(self._cls,"__le__")
            delattr(self._cls,"__ge__")
