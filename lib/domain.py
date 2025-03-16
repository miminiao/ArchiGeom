from abc import ABC
from typing import Callable,Self
from lib.utils import Constant,SupportsCompareWithTolerance,SupportsCompare,ListTool

class Domain[T:SupportsCompare](ABC,SupportsCompareWithTolerance):
    """区间基类"""
    _compare=Constant.DEFAULT.compare_val
    """区间端点值的比较方法"""

    def __init__(self) -> None: ...

class Domain1d[T:SupportsCompare](Domain[T]):
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
    def __eq__(self,other:Self)->bool:
        if not isinstance(other,Domain1d): return False
        if self.is_empty() and other.is_empty():return True
        return self._compare(self.l,other.l)==0 and self._compare(self.r,other.r)==0 and self.value==other.value
    def __add__(self,other:"Self|MultiDomain1d")->"Self|MultiDomain1d":
        """区间合并"""
        if isinstance(other,MultiDomain1d):
            return MultiDomain1d([self])+other
        if self.is_empty(): return other
        if other.is_empty(): return self
        # 没有交集就返回两段独立的
        if not self.is_overlap(other,True):
            return MultiDomain1d([self,other]) if self.l<other.l else MultiDomain1d([other,self])
        # 有交集就切成三段(minl,maxl,vl),(maxl,minr,vmid),(minr,maxr,vr)
        minl,maxl,vl=(self.l,other.l,self.value) if self.l<other.l else (other.l,self.l,other.value)
        minr,maxr,vr=(self.r,other.r,other.value) if self.r<other.r else (other.r,self.r,self.value)
        if self.value==other.value: 
            return Domain1d(minl,maxr,self.value)
        vmid=max(self.value,other.value)
        doms=[Domain1d(minl,maxl,vl),Domain1d(maxl,minr,vmid),Domain1d(minr,maxr,vr)]
        # 合并value相同的部分
        res=[]
        for dom in doms:
            if dom.is_empty(): continue
            if len(res)==0 or dom.value!=res[-1].value:
                res.append(dom)
            else: res[-1].r=dom.r
        return res[0] if len(res)==1 else MultiDomain1d(res)
    def __sub__(self,other:"Self|MultiDomain1d")->"Self|MultiDomain1d":
        """区间求差，无视value。为空返回empty区间"""
        if isinstance(other,MultiDomain1d):
            return MultiDomain1d([self])-other
        if not self.is_overlap(other): return self
        d1=Domain1d(self.l,other.l,self.value)
        d2=Domain1d(other.r,self.r,self.value)
        if d1.is_empty():return d2
        if d2.is_empty():return d1
        return MultiDomain1d([d1,d2])
    def __mul__(self,other:"Self|MultiDomain1d")->Self:
        """区间求交。交集value取小。为空返回empty区间"""
        if isinstance(other,MultiDomain1d):
            return MultiDomain1d([self])*other
        if self.is_empty() or other.is_empty(): return Domain1d(1,0,self.value) #EMPTY
        return Domain1d(max(self.l,other.l),min(self.r,other.r),min(self.value,other.value))
    def is_overlap(self,other:"Self|MultiDomain1d",include_endpoints:bool=True)->bool:
        """判断区间是否重叠，当include_endpoints时允许端点重叠"""
        if isinstance(other,MultiDomain1d):
            return other.is_overlap(self)
        if self.is_empty() or other.is_empty(): return False
        base=0 if include_endpoints else 1
        return self._compare(self.r,other.l)>=base and self._compare(other.r,self.l)>=base
    def is_empty(self)->bool:
        """判断区间是否为空，l==r的情况也返回True"""
        return self._compare(self.l,self.r)>=0
    def copy(self,value:T=None)->"Self":
        """返回与当前区间相同、值为value的区间"""
        return Domain1d(self.l,self.r,value or self.value)
    def contains_point(self,point:float,include_endpoints:bool=True)->bool:
        """点在区间范围内"""
        base=0 if include_endpoints else 1
        return self._compare(point,self.l)>=base and self._compare(self.r,point)>=base
    
class MultiDomain1d[T:SupportsCompare](Domain[T]):
    """描述一组Domain1d"""
    def __init__(self,items:list[Domain1d]):
        self.items=items
        self.items.sort(key=lambda domain:domain.l)
    def __eq__(self,other:Self)->bool:
        if not isinstance(other,MultiDomain1d): return False
        if len(self.items)!=len(other.items): return False
        for a,b in zip(self.items,other.items):
            if a!=b: return False
        return True
    def __repr__(self) -> str:
        return f"[{','.join([str(item) for item in self.items])}]"
    def _scan(self,other:Self,get_next_value:Callable[[T,T],T])->Self:
        """扫描线处理区间交并差"""
        endpoints=[]
        for dom in self.items+other.items: endpoints.extend([dom.l,dom.r])
        endpoints=ListTool.sort_and_dedup(endpoints,compare_func=self._compare)
        # 从左到右扫描，ij记录下一个与扫描线相交的区间index
        res=[]
        i,j=0,0
        for event_pt in endpoints:
            i_value,j_value=None,None
            # 扫描到区间内就取区间的value，扫描到右端点就离开此区间
            if i<len(self.items) and self._compare(event_pt,self.items[i].r)==0:
                i+=1
            if i<len(self.items) and self._compare(event_pt,self.items[i].l)>=0:
                i_value=self.items[i].value
            if j<len(other.items) and self._compare(event_pt,other.items[j].r)==0:
                j+=1
            if j<len(other.items) and self._compare(event_pt,other.items[j].l)>=0:
                j_value=other.items[j].value
            # next_value: 以扫描线为左端点的区间value
            next_value=get_next_value(i_value,j_value)
            # value和上一个区间不同，就新开一个区间
            if len(res)==0:
                res.append(Domain1d(event_pt,None,next_value))
            elif next_value!=res[-1].value:
                res[-1].r=event_pt
                res.append(Domain1d(event_pt,None,next_value))
        res=[dom for dom in res if dom.value is not None]
        return res
    def _value_for_add(self,i:T,j:T)->T:
        if i is None and j is None: return None
        elif i is None: return j
        elif j is None: return i
        else: return max(i,j)
    def _value_for_sub(self,i:T,j:T)->T:
        if i is None: return None
        elif j is not None: return None
        else: return i
    def _value_for_mul(self,i:T,j:T)->T:
        if i is None or j is None: return None
        else: return min(i,j)
    def __add__(self,other:Domain1d|Self)->Domain1d|Self:
        """多区间合并"""
        if isinstance(other,Domain1d):
            if other.is_empty():return self
            other=MultiDomain1d([other])
        res=self._scan(other,self._value_for_add)
        if len(res)>1: return MultiDomain1d(res)
        else: return res[0]
    def __sub__(self,other:Domain1d|Self)->Domain1d|Self:
        """多区间求差"""
        if isinstance(other,Domain1d):
            if other.is_empty():return self
            other=MultiDomain1d([other])
        res=self._scan(other,self._value_for_sub)
        if len(res)>1: return MultiDomain1d(res)
        elif len(res)==1: return res[0]
        else: return Domain1d(1,0,0) #EMPTY
    def __mul__(self,other:Domain1d|Self)->Domain1d|Self:
        """多区间求交"""
        if isinstance(other,Domain1d):
            if other.is_empty():return self
            other=MultiDomain1d([other])
        res=self._scan(other,self._value_for_mul)
        if len(res)>1: return MultiDomain1d(res)
        elif len(res)==1: return res[0]
        else: return Domain1d(1,0,0) #EMPTY
    def is_empty(self)->bool:
        return len(self.items)==0 or len(self.items)==1 and self.items[0].is_empty()
    def is_overlap(self,other:Domain1d|Self, include_endpoints:bool=True)->bool:
        """判断区间是否重叠，当include_endpoints时允许端点重叠"""
        if isinstance(other,Domain1d):
            other=MultiDomain1d([other])
        i,j=0,0
        while True:
            while i<len(self.items) and self._compare(self.items[i].r,other.items[j].l)<0: i+=1
            if i==len(self.items): return False
            while j<len(other.items) and self._compare(other.items[j].r,self.items[i].l)<0: j+=1
            if j==len(other.items): return False
            if self.items[i].is_overlap(other.items[j],include_endpoints=include_endpoints):
                return True
    def copy(self,value:float=None)->Self:
         """返回与当前区间相同、高度为value的区间"""
         return MultiDomain1d([Domain1d(item.l,item.r,value or item.value) for item in self.items])