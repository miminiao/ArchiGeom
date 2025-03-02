from typing import Callable
from abc import ABC,abstractmethod
from lib.utils import Constant

class Domain[T](ABC):
    """区间基类."""
    compare=lambda x,y:0
    _compare_stack=[compare]
    def __init__(self) -> None: ...
    @property
    def const(self)->Constant:
        return Constant.get()
    @classmethod
    def push_compare(cls,compare:Callable[[T,T],int]=None)->None:
        """定义当前上下文的compare函数

        Args:
            compare (Callable[[T,T],int], optional): value的比较函数，合并时保留较大的. Defaults to 0.
        """
        cls._compare_stack.append(cls.compare)
        cls.compare=compare
    @classmethod
    def pop_compare(cls)->Callable[[T,T],int]:
        return cls._compare_stack.pop()
    
class Domain1d[T](Domain):
    def __init__(self,l:float,r:float,value:T) -> None:
        """描述一个带有附加值的一维区间.

        Args:
            l (float): 区间下限
            r (float): 区间上限
            value (T): 附加值
        """
        super().__init__()
        self.l=l
        self.r=r
        self.value=value
    def __repr__(self) -> str:
        return f"({self.l},{self.r},{self.value})"
    def __eq__(self,other:"Domain1d")->bool:
        if not isinstance(other,Domain1d): return False
        if self.is_empty() and other.is_empty():return True
        tol_comp=self.const.get_compare_func()
        return tol_comp(self.l,other.l)==0 and tol_comp(self.r,other.r)==0 and self.compare(self.value,other.value)==0
    def __add__(self,other:"Domain1d| MultiDomain1d")->"Domain1d | MultiDomain1d":
        """区间合并"""
        if isinstance(other,MultiDomain1d):
            return MultiDomain1d([self])+other
        if self.is_empty(): return other
        if other.is_empty(): return self
        # 没有交集就返回两段独立的
        if not self.is_overlap(other,True):
            return MultiDomain1d([self,other])
        # 有交集就切成三段(minl,maxl,v_minl),(maxl,minr,v),(minr,maxr,v_maxr)
        minl,maxl,v_minl=(self.l,other.l,self.value) if self.l<other.l else (other.l,self.l,other.value)
        minr,maxr,v_maxr=(self.r,other.r,other.value) if self.r<other.r else (other.r,self.r,self.value)
        match self.compare(self.value,other.value):
            case 0: return Domain1d(minl,maxr,self.value)
            case 1: v=self.value
            case -1: v=other.value
        res=[Domain1d(maxl,minr,v)]
        # 合并value相同的部分
        if self.compare(v_minl,v)==0:
            res[0].l=minl
        else:
            dom=Domain1d(minl,maxl,v_minl)
            if not dom.is_empty():
                res.insert(0,dom)
        if self.compare(v_maxr,v)==0:
            res[-1].r=maxr
        else:
            dom=Domain1d(minr,maxr,v_maxr)
            if not dom.is_empty():
                res.append(dom)
        return res[0] if len(res)==1 else MultiDomain1d(res)
    def __sub__(self,other:"Domain1d | MultiDomain1d")->"Domain1d | MultiDomain1d":
        """区间求差，无视高矮。为空返回empty区间"""
        if isinstance(other,MultiDomain1d):
            return MultiDomain1d([self])-other
        if not self.is_overlap(other): return self
        d1=Domain1d(self.l,other.l,self.value)
        d2=Domain1d(other.r,self.r,self.value)
        if d1.is_empty():return d2
        if d2.is_empty():return d1
        return MultiDomain1d([d1,d2])
    def __mul__(self,other:"Domain1d | MultiDomain1d")->"Domain1d":
        """区间求交。为空返回empty区间"""
        if isinstance(other,MultiDomain1d):
            return MultiDomain1d([self])*other
        if self.is_empty() or other.is_empty(): return Domain1d(1,0,self.value) #EMPTY
        # 根据value判断
        match self.compare(self.value,other.value):
            case 0|1: return Domain1d(max(self.l,other.l),min(self.r,other.r),self.value)
            case -1: return Domain1d(max(self.l,other.l),min(self.r,other.r),other.value) 
    def is_overlap(self,other:"Domain1d | MultiDomain1d",include_endpoints:bool=True)->bool:
        """判断区间是否重叠，当include_endpoints时允许端点重叠"""
        if isinstance(other,MultiDomain1d):
            return other.is_overlap(self)
        if self.is_empty() or other.is_empty(): return False
        tol_comp=self.const.get_compare_func()
        if include_endpoints:
            return tol_comp(self.r,other.l)>=0 and tol_comp(other.r,self.l)>=0
        else:
            return tol_comp(self.r,other.l)==1 and tol_comp(other.r,self.l)==1
    def is_empty(self)->bool:
        """判断区间是否为空，l==r的情况也返回True"""
        tol_comp=self.const.get_compare_func()
        return tol_comp(self.l,self.r)>=0
    def copy(self,value:T=None)->"Domain1d":
        """返回与当前区间相同、高度为h的区间"""
        return Domain1d(self.l,self.r,value or self.value)
class MultiDomain1d:
    """描述一组Domain1d"""
    def __init__(self,items:list[Domain1d]):
        self.items=items
        self.items.sort(key=lambda domain:domain.l)
    def __eq__(self,other:"MultiDomain1d")->bool:
        if not isinstance(other,MultiDomain1d): return False
        if len(self.items)!=len(other.items): return False
        for a,b in zip(self.items,other.items):
            if a!=b: return False
        return True
    def __repr__(self) -> str:
        return f"[{','.join([str(item) for item in self.items])}]"
    def __add__(self,other:"Domain1d | MultiDomain1d")->"Domain1d | MultiDomain1d":
        """多区间合并"""
        if isinstance(other,Domain1d):
            if other.is_empty():return self
            other_items=[other] 
        else: other_items=other.items
        res=[Domain1d(item.l,item.r,item.value) for item in self.items+other_items]
        res.sort(key=lambda item:item.r)
        i=len(res)-2
        while i>=0:
            if res[i].is_overlap(res[i+1],True):
                sum=res[i]+res[i+1]
                res.remove(res[i+1])
                res.remove(res[i])
                if isinstance(sum,Domain1d):
                    res.insert(i,sum)
                else:
                    for j in range(len(sum.items)-1,-1,-1):
                        res.insert(i,sum.items[j])
            i-=1
        if len(res)>1: return MultiDomain1d(res)
        else: return res[0]
    def __sub__(self,other:"Domain1d | MultiDomain1d")->"Domain1d | MultiDomain1d":
        """多区间求差"""
        if isinstance(other,Domain1d):
            other=MultiDomain1d([other])
        res=[Domain1d(item.l,item.r,item.value) for item in self.items]
        i=0
        while i<len(res):
            subtracted=False
            for dom_j in other.items:
                overlapped=res[i].is_overlap(dom_j,False)
                if overlapped:
                    diff=res[i]-dom_j
                    if isinstance(diff,Domain1d):
                        res[i]=diff
                    else:
                        res[i]=diff.items[0]
                        res.insert(i+1,diff.items[1])
                    subtracted=True
                elif subtracted: break
                else: continue
            if res[i].is_empty():
                res.remove(res[i])
            else: i+=1
        if len(res)>1: return MultiDomain1d(res)
        elif len(res)==1: return res[0]
        else: return Domain1d(1,0,0) #EMPTY
    def __mul__(self,other:"Domain1d | MultiDomain1d")->"Domain1d | MultiDomain1d":
        """多区间求交"""
        if isinstance(other,Domain1d):
            other=MultiDomain1d([other])
        res=[]
        for i in self.items:
            intersected=False
            for j in other.items:
                intersection=i*j
                if not intersection.is_empty():
                    res.append(intersection)
                    intersected=True
                elif intersected: break
                else: continue
        if len(res)>1: return MultiDomain1d(res)
        elif len(res)==1: return res[0]
        else: return Domain1d(1,0,0) #EMPTY
    def is_empty(self)->bool:
        return len(self.items)==0 or len(self.items)==1 and self.items[0].is_empty()
    def is_overlap(self,other:"Domain1d",include_endpoints:bool=True)->bool:
        """判断区间是否重叠，当include_endpoints时允许端点重叠"""
        if isinstance(other,Domain1d):
            other=MultiDomain1d([other])
        for i in self.items:
            for j in other.items:
                if i.is_overlap(j,include_endpoints=include_endpoints):
                    return True
        return False
    def copy(self,value:float=None)->"MultiDomain1d":
         """返回与当前区间相同、高度为v的区间"""
         return MultiDomain1d([Domain1d(item.l,item.r,value or item.value) for item in self.items])