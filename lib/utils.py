"""通用工具类"""

import json
import pathlib
import math
from time import time,sleep
from typing import Any,Callable,Self,Protocol
from functools import wraps

# -----------------------------------------------------------------------------

class Timer:
    """计时器.
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
        self.start=time()
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self._enabled:
            disc=self._tag
            t=time()-self.start
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
TWO_PI=math.pi*2

class Constant:
    """全局常量类，默认值从配置文件读取: ./env.json['CONSTANTS']
    
    Args: 
    Example:
    - 使用类的实例: 
    >>> Constant.DEFAULT.TOL_DIST
    1e-2

    - 使用类属性:
    >>> Constant.TOL_DIST
    1e-2

    - 使用上下文管理器: 
    >>> with Constant(tol_dist=1e-1):
    ...     print(Constant.TOL_DIST)
    1e-1
    
    - 使用比较函数:
    >>> Constant.compare_dist(1.000,1.001)
    0
    """
    _arg_names=["MAX_VAL","TOL_VAL","TOL_DIST","TOL_AREA","TOL_ANG"]
    _stack:list[Self]=[]
    DEFAULT:Self=None

    MAX_VAL,TOL_ANG,TOL_AREA,TOL_DIST,TOL_VAL=[None]*5

    def __init__(self,
                 max_val:float=None,
                 tol_val:float=None,
                 tol_dist:float=None,
                 tol_area:float=None,
                 tol_ang:float=None,
                 ) -> None:
        """自定义常量.

        Args:
            max_val (float, optional): 正无穷. Defaults to DEFAULT.MAX_VAL.
            tol_val (float, optional): 浮点容差. Defaults to DEFAULT.TOL_VAL.
            tol_dist (float, optional): 距离容差. Defaults to DEFAULT.TOL_DIST.
            tol_area (float, optional): 面积容差. Defaults to DEFAULT.TOL_AREA.
            tol_ang (float, optional): 角度容差. Defaults to DEFAULT.TOL_ANG.
        """
        self.MAX_VAL=max_val or self.DEFAULT.MAX_VAL
        self.TOL_VAL=tol_val or self.DEFAULT.TOL_VAL
        self.TOL_DIST=tol_dist or self.DEFAULT.TOL_DIST
        self.TOL_AREA=tol_area or self.DEFAULT.TOL_AREA
        self.TOL_ANG=tol_ang or self.DEFAULT.TOL_ANG
    @classmethod
    def compare_val(cls,x:float,y:float)->int:
        if abs(x-y)<cls.TOL_VAL: return 0
        elif x>y: return 1
        else: return -1
    @classmethod        
    def compare_dist(cls,x:float,y:float)->int:
        if abs(x-y)<cls.TOL_DIST: return 0
        elif x>y: return 1
        else: return -1
    @classmethod
    def compare_area(cls,x:float,y:float)->int:
        if abs(x-y)<cls.TOL_AREA: return 0
        elif x>y: return 1
        else: return -1
    @classmethod
    def compare_ang(cls,x:float,y:float,periodic:bool=True)->int:
        """periodic==True时直接比较；periodic==False时先将x和y换算到[0,2pi)范围"""
        if not periodic:
            x%=TWO_PI
            y%=TWO_PI
            if TWO_PI-abs(x-y)<cls.TOL_ANG: return 0         
        if abs(x-y)<cls.TOL_ANG: return 0
        elif x>y: return 1
        else: return -1
    def __enter__(self):
        Constant._push(self)
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        Constant._pop()
    @classmethod
    def get(cls)->Self:
        return cls._stack[-1]
    @classmethod
    def _init_default(cls)-> Self:
        with open(pathlib.Path(__file__).parent.parent/"env.json",'r') as f:
            js=json.load(f)["CONSTANTS"]
        args=[float(js[name]) for name in cls._arg_names]
        cls.DEFAULT=cls(*args)
        cls._push(cls.DEFAULT)
    @classmethod
    def _push(cls,item:Self)->None:
        cls._stack.append(item)
        for arg_name in cls._arg_names:
            setattr(cls,arg_name,getattr(item,arg_name))
    @classmethod
    def _pop(cls)->Self:
        cls._stack.pop()
        for arg_name in cls._arg_names:
            setattr(cls,arg_name,getattr(cls._stack[-1],arg_name))

Constant._init_default()

# -----------------------------------------------------------------------------

class ListTool:
    @staticmethod
    def sort_and_dedup(a: list[float],
                       tol:float=None,
                       compare_func:Callable[[float,float],int]=None
    ) -> list[float]:
        """排序并去除重复float元素"""
        if len(a)==0: return
        if compare_func is None:
            tol=tol or Constant.TOL_VAL
            compare_func=lambda x,y:0 if abs(x-y)<tol else 1
        tmp=sorted(a)
        res=[tmp[0]]
        for _,x in enumerate(tmp,start=1):
            if compare_func(x,res[-1])!=0:
                res.append(x)
        return res
    @staticmethod
    def search_value(a: list, x: float, key:Callable[[Any],float]=None) -> tuple[bool, int]:
        """在a(sorted)中二分查找x的index; 找不到则返回应当插入的位置.

        Args:
            a (list): 按照key排好序的list.
            x (float): 查找的key值.
            key (Callable[[Any],float], optional): _description_. Defaults to None.

        Returns:
            bool: 是否查找成功
            int: 所在的index(找到了) / 应该插入到的位置(没找到)
        """
        tol=Constant.get().TOL_VAL
        if key is None:
            key = lambda x: x
        l, r = 0, len(a) - 1
        while l <= r:
            m = (l + r) // 2
            if abs(x - key(a[m])) < tol:
                return True, m
            elif x < key(a[m]):
                r = m - 1
            else:
                l = m + 1
        return False, l
    @staticmethod
    def find_first(a:list, cond:Callable[[Any],bool])->int:
        for i,item in enumerate(a):
            if cond(item): return i
        return -1

class StopRetry(BaseException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.cause=args[0]

def retry(max_times:int=10,interval:float=0):
    """Decorator: 失败后重试, 超过上限抛出错误.

    Example:
    >>> @retry()
    >>> def foo(): raise RuntimeError("bar")
    ...
    RuntimeError: bar

    Args:
        max_times (int, optional): 最大次数. Defaults to 10.
        interval (float, optional): 间隔等待时间/秒. Defaults to 0.
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
                    print(e.__class__.__name__,":",e,". Retrying...")
                    last_err=e
                    retry_count+=1
                    if interval>0: sleep(interval)
            raise last_err
        return wrapper
    return decorator

class SupportsCompareWithTolerance(Protocol):
    _compare:Callable[[float,float],int]

class SupportsCompare(Protocol):
    def __lt__(self,other)->bool: ...
    def __gt__(self,other)->bool: ...
    def __eq__(self,other)->bool: ...
    def __le__(self,other)->bool: ...
    def __ge__(self,other)->bool: ...
    def __ne__(self,other)->bool: ...

class ComparerInjector[T]:
    """比较器.
    - 作为上下文管理器: 
    >>> compare_wall=lambda x,y:Constant.compare_dist(x.width,y.width)
    >>> with Comparer(Wall,compare_wall,override_ops=True) as comparer:
    ...     print(Wall(width=100)>Wall(width=50))
    True
    """
    def __init__(self,cls:type[T],func:Callable[[T,T],int],override_ops:bool=False) -> None:
        self._cls=cls
        self._func=func
        self._override_ops=override_ops

    def __enter__(self):
        if "_compare" in self._cls.__dict__:
            self._previous_func=self._cls._compare
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
        self._cls._compare=func
        if self._override_ops:
            self._cls.__lt__=lambda self,other: func(self,other)<0
            self._cls.__gt__=lambda self,other: func(self,other)>0
            self._cls.__eq__=lambda self,other: func(self,other)==0
            self._cls.__le__=lambda self,other: func(self,other)<=0
            self._cls.__ge__=lambda self,other: func(self,other)>=0
            self._cls.__ne__=lambda self,other: func(self,other)!=0
            self._cls.__hash__=id(self)
    def _eject(self)->None:
        delattr(self._cls,"_compare")
        if self._override_ops:
            delattr(self._cls,"__lt__")
            delattr(self._cls,"__gt__")
            delattr(self._cls,"__eq__")
            delattr(self._cls,"__le__")
            delattr(self._cls,"__ge__")
            delattr(self._cls,"__ne__")