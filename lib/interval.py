from abc import ABC
from typing import Callable,Self
from lib.utils import Constant,SupportsCompareFunc,SupportsCompareOps,ListTool

class Interval[T:SupportsCompareOps](ABC,SupportsCompareFunc):
    """区间基类"""
    _cmp=Constant.cmp_dist
    """区间端点值的比较方法"""

    def __init__(self) -> None: ...

class Interval1d[T:SupportsCompareOps](Interval[T]):
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
        if not isinstance(other,Interval1d): return False
        if self.is_empty() and other.is_empty():return True
        return self._cmp(self.l,other.l)==0 and self._cmp(self.r,other.r)==0 and self.value==other.value
    def __add__(self,other:"Self|MultiInterval1d")->"Self|MultiInterval1d":
        """区间合并"""
        if isinstance(other,MultiInterval1d):
            return MultiInterval1d([self])+other
        if self.is_empty(): return other
        if other.is_empty(): return self
        # 没有交集就返回两段独立的
        if not self.is_overlap(other,True):
            return MultiInterval1d([self,other]) if self.l<other.l else MultiInterval1d([other,self])
        # 有交集就切成三段(minl,maxl,vl),(maxl,minr,vmid),(minr,maxr,vr)
        minl,maxl,vl=(self.l,other.l,self.value) if self.l<other.l else (other.l,self.l,other.value)
        minr,maxr,vr=(self.r,other.r,other.value) if self.r<other.r else (other.r,self.r,self.value)
        if self.value==other.value: 
            return Interval1d(minl,maxr,self.value)
        vmid=max(self.value,other.value)
        intvs=[Interval1d(minl,maxl,vl),Interval1d(maxl,minr,vmid),Interval1d(minr,maxr,vr)]
        # 合并value相同的部分
        res=[]
        for intv in intvs:
            if intv.is_empty(): continue
            if len(res)==0 or intv.value!=res[-1].value:
                res.append(intv)
            else: res[-1].r=intv.r
        return res[0] if len(res)==1 else MultiInterval1d(res)
    def __sub__(self,other:"Self|MultiInterval1d")->"Self|MultiInterval1d":
        """区间求差，无视value。为空返回empty区间"""
        if isinstance(other,MultiInterval1d):
            return MultiInterval1d([self])-other
        if not self.is_overlap(other): return self
        d1=Interval1d(self.l,other.l,self.value)
        d2=Interval1d(other.r,self.r,self.value)
        if d1.is_empty():return d2
        if d2.is_empty():return d1
        return MultiInterval1d([d1,d2])
    def __mul__(self,other:"Self|MultiInterval1d")->Self:
        """区间求交。交集value取小。为空返回empty区间"""
        if isinstance(other,MultiInterval1d):
            return MultiInterval1d([self])*other
        if self.is_empty() or other.is_empty(): return Interval1d(1,0,self.value) #EMPTY
        return Interval1d(max(self.l,other.l),min(self.r,other.r),min(self.value,other.value))
    def is_overlap(self,other:"Self|MultiInterval1d",include_endpoints:bool=True)->bool:
        """判断区间是否重叠，当include_endpoints时允许端点重叠"""
        if isinstance(other,MultiInterval1d):
            return other.is_overlap(self)
        if self.is_empty() or other.is_empty(): return False
        base=0 if include_endpoints else 1
        return self._cmp(self.r,other.l)>=base and self._cmp(other.r,self.l)>=base
    def is_empty(self)->bool:
        """判断区间是否为空，l==r的情况也返回True"""
        return self._cmp(self.l,self.r)>=0
    def copy(self,value:T=None)->"Self":
        """返回与当前区间相同、值为value的区间"""
        return Interval1d(self.l,self.r,value or self.value)
    def contains_point(self,point:float,include_endpoints:bool=True)->bool:
        """点在区间范围内"""
        base=0 if include_endpoints else 1
        return self._cmp(point,self.l)>=base and self._cmp(self.r,point)>=base
    @classmethod
    def union(cls,intervals:list[Self],ignore_value:bool=True)->"MultiInterval1d[T]":
        """多个区间合并。ignore_value时，顺序遍历合并；否则二路归并"""
        if len(intervals)==0: return None
        intervals=intervals[:]
        if ignore_value:
            intervals.sort(key=lambda intv:intv.l)
            res=[intervals[0]]
            for intv in intervals:
                if cls._cmp(intv.l,res[-1].r)<=0:  # 有重叠部分就延长
                    res[-1].r=max(res[-1].r,intv.r)
                else:  #　没有重叠部分就新加一段
                    res.append(intv)
            res=list(filter(lambda x:not x.is_empty(),res))
            return MultiInterval1d(res)
        else:
            return cls._merge_union(intervals)
    @classmethod
    def _merge_union(cls,intvs:list[Self])->"MultiInterval1d":
        count=len(intvs)
        if count<=1: return MultiInterval1d(intvs)
        else: 
            l=cls._merge_union(intvs[:count//2])
            r=cls._merge_union(intvs[count//2:])
            res=l+r
            return res
    
class MultiInterval1d[T:SupportsCompareOps](Interval[T]):
    """描述一组互无交集的Interval1d"""
    def __init__(self,items:list[Interval1d]):
        self._items=sorted(items,key=lambda interval:interval.l)
    def __repr__(self) -> str:
        return f"[{','.join([str(item) for item in self._items])}]"
    def __iter__(self):
        return iter(self._items)
    def __len__(self):
        return len(self._items)
    def __getitem__(self,index):
        return self._items[index]
    def __eq__(self,other:Self)->bool:
        if not isinstance(other,MultiInterval1d): return False
        if len(self)!=len(other): return False
        for a,b in zip(self._items,other._items):
            if a!=b: return False
        return True
    def _scan(self,other:Self,get_next_value:Callable[[T,T],T])->Self:
        """扫描线处理区间交并差"""
        endpoints=[]
        for intv in self._items+other._items: endpoints.extend([intv.l,intv.r])
        endpoints=ListTool.distinct(endpoints,cmp_func=self._cmp)
        # 从左到右扫描，ij记录下一个与扫描线相交的区间index
        res=[]
        i,j=0,0
        for event_pt in endpoints:
            i_value,j_value=None,None
            # 扫描到区间内就取区间的value，扫描到右端点就离开此区间
            if i<len(self) and self._cmp(event_pt,self[i].r)==0:
                i+=1
            if i<len(self) and self._cmp(event_pt,self[i].l)>=0:
                i_value=self[i].value
            if j<len(other) and self._cmp(event_pt,other[j].r)==0:
                j+=1
            if j<len(other) and self._cmp(event_pt,other[j].l)>=0:
                j_value=other[j].value
            # next_value: 以扫描线为左端点的区间value
            next_value=get_next_value(i_value,j_value)
            # value和上一个区间不同，就新开一个区间
            if len(res)==0:
                res.append(Interval1d(event_pt,None,next_value))
            elif next_value!=res[-1].value:
                res[-1].r=event_pt
                res.append(Interval1d(event_pt,None,next_value))
        res=[intv for intv in res if intv.value is not None]
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
    def __add__(self,other:Interval1d|Self)->Interval1d|Self:
        """多区间合并"""
        if isinstance(other,Interval1d):
            if other.is_empty():return self
            other=MultiInterval1d([other])
        res=self._scan(other,self._value_for_add)
        if len(res)>1: return MultiInterval1d(res)
        else: return res[0]
    def __sub__(self,other:Interval1d|Self)->Interval1d|Self:
        """多区间求差"""
        if isinstance(other,Interval1d):
            if other.is_empty():return self
            other=MultiInterval1d([other])
        res=self._scan(other,self._value_for_sub)
        if len(res)>1: return MultiInterval1d(res)
        elif len(res)==1: return res[0]
        else: return Interval1d(1,0,0) #EMPTY
    def __mul__(self,other:Interval1d|Self)->Interval1d|Self:
        """多区间求交"""
        if isinstance(other,Interval1d):
            if other.is_empty():return self
            other=MultiInterval1d([other])
        res=self._scan(other,self._value_for_mul)
        if len(res)>1: return MultiInterval1d(res)
        elif len(res)==1: return res[0]
        else: return Interval1d(1,0,0) #EMPTY
    def is_empty(self)->bool:
        for i in self: 
            if not i.is_empty(): return False
        return True
    def is_overlap(self,other:Interval1d|Self, include_endpoints:bool=True)->bool:
        """判断区间是否重叠，当include_endpoints时允许端点重叠"""
        if isinstance(other,Interval1d):
            other=MultiInterval1d([other])
        i,j=0,0
        while True:
            while i<len(self) and self._cmp(self[i].r,other[j].l)<0: i+=1
            if i==len(self): return False
            while j<len(other) and self._cmp(other[j].r,self[i].l)<0: j+=1
            if j==len(other): return False
            if self[i].is_overlap(other[j],include_endpoints=include_endpoints):
                return True
    def copy(self,value:float=None)->Self:
         """返回与当前区间相同、高度为value的区间"""
         return MultiInterval1d([Interval1d(item.l,item.r,value or item.value) for item in self])