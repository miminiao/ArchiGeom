"""通用工具类"""

import json
import pathlib
import math
from time import time,sleep
from typing import Any,Callable,Self
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
    def __init__(self,func=None,tag:str="") -> None:
        self._func=func
        self._tag=tag
    def __enter__(self):
        self.start=time()
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self._enabled:
            print(self._tag,time()-self.start)
    def __call__(self,*args,**kwargs):
        t0=time()
        ret=self._func(*args,**kwargs)
        if self._enabled:
            print(self._tag+self._func.__name__,time()-t0)
        return ret
    @classmethod
    def enable(cls): cls._enabled=True
    @classmethod
    def disable(cls): cls._enabled=False

# -----------------------------------------------------------------------------

class Constant:
    """全局常量类.

    默认值从配置文件读取: ./env.json['CONSTANTS']

    - 使用类的实例: 
    >>> from utils import DEFAULT_CONSTANT as const
    >>> print(const.TOL_DIST)
    1e-2
    >>> from utils import Constant
    >>> print(Constant.get().TOL_DIST)
    1e-2

    - 使用上下文管理器: 
    >>> with Constant(tol_dist=1e-3) as const:
    ...     print(const.TOL_DIST, const.MAX_VAL)
    1e-3
    """
    _arg_names=["MAX_VAL","TOL_VAL","TOL_DIST","TOL_AREA","TOL_ANG"]
    _tags={}
    _stack=[]
    _DEFAULT:Self=None

    def __init__(self,
                 max_val:float=None,
                 tol_val:float=None,
                 tol_dist:float=None,
                 tol_area:float=None,
                 tol_ang:float=None,
                 tag:str=None,
                 ) -> None:
        """自定义常量.

        Args:
            max_val (float, optional): 正无穷. Defaults to DEFAULT.MAX_VAL.
            tol_val (float, optional): 浮点容差. Defaults to DEFAULT.TOL_VAL.
            tol_dist (float, optional): 距离容差. Defaults to DEFAULT.TOL_DIST.
            tol_area (float, optional): 面积容差. Defaults to DEFAULT.TOL_AREA.
            tol_ang (float, optional): 角度容差. Defaults to DEFAULT.TOL_ANG.
            tag (str, optional): 唯一标签名; 同名将会替换. Defaults to None.
        """
        if tag is not None: 
            self._register(tag)
        self.tag=tag
        self.MAX_VAL=max_val or self._DEFAULT.MAX_VAL
        self.TOL_VAL=tol_val or self._DEFAULT.TOL_VAL
        self.TOL_DIST=tol_dist or self._DEFAULT.TOL_DIST
        self.TOL_AREA=tol_area or self._DEFAULT.TOL_AREA
        self.TOL_ANG=tol_ang or self._DEFAULT.TOL_ANG
    def __enter__(self):
        self.push(self)
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        _=self.pop()
    def _register(self,tag:str,update:bool=False):
        if not isinstance(tag,str): raise TypeError
        Constant._tags[tag]=self
    @classmethod
    def get(cls,tag:str=None)->Self:
        """按标签名获取实例. 默认返回栈顶实例."""
        if tag is None: return cls._stack[-1]
        else: return cls._tags[tag]
    def _comp(self,a:float,b:float,tol_type:str="TOL_DIST")->int:
        if tol_type=="TOL_ANG":  # 对于角度，额外判断0与2pi
            if 2*math.pi-abs(a-b)<self.TOL_ANG: return 0 
        if abs(a-b)<getattr(self,tol_type): return 0
        elif a>b: return 1
        else: return -1
    def get_comp_func(self,tol_type:str="TOL_DIST")->Callable[[float,float],int]:
        """获取比较函数.

        Args:
            tol_type (str, optional): 常量名称. Defaults to "TOL_VAL".

        Returns:
            Callable[[float,float],int]: eq->0, gt->1, lt->-1.
        """
        if tol_type not in self._arg_names: return ValueError(f"Tolerance type '{tol_type}' not supported.")
        return lambda a,b:self._comp(a,b,tol_type)
    @classmethod
    def push(cls,instance:Self)->None:
        cls._stack.append(instance)
    @classmethod
    def pop(cls)->Self:
        return cls._stack.pop()
    @classmethod
    def _init_default(cls)-> Self:
        with open(pathlib.Path(__file__).parent.parent/"env.json",'r') as f:
            js=json.load(f)["CONSTANTS"]
        args=[float(js[name]) for name in cls._arg_names]
        cls._DEFAULT=cls(*args,tag="DEFAULT")
        cls.push(cls._DEFAULT)
        return cls._DEFAULT

DEFAULT_CONSTANT=Constant._init_default()

# -----------------------------------------------------------------------------

class ListTool:
    @staticmethod
    def sort_and_overkill(a: list[float]) -> None:
        """排序并去除重复float元素"""
        tol=Constant.get().TOL_VAL
        if len(a) <= 1:
            return
        a.sort()
        for i in range(len(a) - 1, 0, -1):
            if a[i] - a[i - 1] < tol:
                del a[i]
    @staticmethod
    def search_value(a: list, x: float, key:Callable[[Any],float]=None) -> tuple[bool, int]:
        """在a(sorted)中二分查找x的index。找不到则返回应当插入的位置

        Args:
            a (list): 按照key排好序的list
            x (float): 查找的key值
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