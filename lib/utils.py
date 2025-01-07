"""通用工具"""

import json
from time import time
from functools import wraps
from typing import Any,Callable

class Timer:
    def __init__(self,name:str="__main__",tag:str="") -> None:
        self.name=name
        self.tag=tag
    def __enter__(self):
        self.start=time()
        return self
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self.name=="__main__":
            print(self.tag,time()-self.start)
    def __call__(self,func):
        @wraps(func)
        def wrapper(*args,**kwargs):
            t0=time()
            ret=func(*args,**kwargs)
            if self.name=="__main__":
                print(self.tag+func.__name__,time()-t0)
            return ret
        return wrapper

class Constant:
    """常量类. 默认值从配置文件读取: lib/constants.json"""
    with open("lib/constants.json",'r') as f:
        js=json.load(f)
        MAX_VAL=float(js["MAX_VAL"]) if "MAX_VAL" in js else float("inf")
        TOL_VAL=float(js["TOL_VAL"])
        TOL_DIST=float(js["TOL_DIST"])
        TOL_AREA=float(js["TOL_AREA"])
        TOL_ANG=float(js["TOL_ANG"])
    _instances={}
    def __init__(self,
                 tag:str,
                 max_val:float=None,
                 tol_val:float=None,
                 tol_dist:float=None,
                 tol_area:float=None,
                 tol_ang:float=None
                 ) -> None:
        """自定义常量.

        Args:
            tag (str): 唯一标签.
            max_val (float, optional): 正无穷. Defaults to constants.json["MAX_VAL"] | float("inf").
            tol_val (float, optional): 浮点容差. Defaults to constants.json["TOL_VAL"].
            tol_dist (float, optional): 距离容差. Defaults to constants.json["TOL_DIST"].
            tol_area (float, optional): 面积容差. Defaults to constants.json["TOL_AREA"].
            tol_ang (float, optional): 角度容差. Defaults to constants.json["TOL_ANG"].
            tol_dist2 (float, READ ONLY): 单位面积容差，用于判断单位向量叉乘. =1-(1-tol_dist)**2.
        """
        if tag in Constant._instances:
            raise ValueError(f"Constant with tag {tag} already exists.")
        self.MAX_VAL=max_val or Constant.MAX_VAL
        self.TOL_VAL=tol_val or Constant.TOL_VAL
        self.TOL_DIST=tol_dist or Constant.TOL_DIST
        self.TOL_AREA=tol_area or Constant.TOL_AREA
        self.TOL_ANG=tol_ang or Constant.TOL_ANG
        self.TOL_DIST2=1-(1-self.TOL_DIST)**2
        Constant._instances[tag]=self
    def scale_by(self,scale_factor:float)->"Constant":
        """缩放当前常量配置"""
        return Constant(self.MAX_VAL,
                        self.TOL_VAL*scale_factor,
                        self.TOL_DIST*scale_factor,
                        self.TOL_AREA*scale_factor,
                        self.TOL_ANG*scale_factor,
                        )
    @classmethod
    def default(cls):
        """返回默认的常量配置"""
        if not hasattr(cls,"DEFAULT"):
            cls.DEFAULT=cls(tag="default")
        return cls.DEFAULT
    @classmethod
    def get_const_config(cls,tag:str)->"Constant":
        return cls._instances[tag]
    def comp(self,a:float,b:float,tol_type:str="TOL_DIST"):
        """比较大小"""
        if abs(a-b)<getattr(self,tol_type): return 0
        elif a>b: return 1
        else: return -1

class ListTool:
    def sort_and_overkill(a: list[float],const:Constant=None) -> None:
        """排序并去除重复float元素"""
        const=const or Constant.default()
        if len(a) <= 1:
            return
        a.sort()
        for i in range(len(a) - 1, 0, -1):
            if a[i] - a[i - 1] < const.TOL_VAL:
                del a[i]

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
        if key is None:
            key = lambda x: x
        l, r = 0, len(a) - 1
        while l <= r:
            m = (l + r) // 2
            if abs(x - key(a[m])) < Constant.default().TOL_VAL:
                return True, m
            elif x < key(a[m]):
                r = m - 1
            else:
                l = m + 1
        return False, l