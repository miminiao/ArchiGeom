import math
import numpy as np
from copy import copy
from typing import Callable
from abc import ABC,abstractmethod
from itertools import groupby

from shapely.geometry import Polygon,Point,box
from shapely.affinity import rotate
from shapely import prepare

from lib.linalg import Vec3d
from lib.domain import Domain1d
from lib.geom import Geom,Node,LineSeg,Arc,Edge,Loop,Poly
from lib.utils import Timer,Constant,ListTool
from lib.index import STRTree,TreeNode,SegmentTree
from lib.building_element import Wall

class GeomAlgo(ABC):
    def __init__(self,const:Constant=None) -> None:
        self.const=const or Constant.default()
    @abstractmethod
    def _preprocess(self)->None:
        Geom.push_const(self.const)
    @abstractmethod
    def get_result(self):
        ...
    @abstractmethod
    def _postprocess(self)->None:
        Geom.pop_const()
class MaxRectAlgo(GeomAlgo):
    def __init__(
            self,
            poly: Poly,
            order: int = 1,
            covered_points: list[list[Node]] = None,
            precision: float = -1.0,
            cut_depth=1,
            const:Constant=None,
        ) -> None:
        """多边形与坐标轴平行的最大内接矩形

        Args:
            poly (Poly): 多边形.
            order (int, optional): 最大矩形的数量. Defaults to 1.
            covered_points (list[list[Node]], optional): 第1..order大矩形必须包含的点坐标，没有则对应位置=None. Defaults to None.
            precision (float, optional): 斜线的xy分割网格尺寸，单位mm；-1.0即不分割. Defaults to -1.0.
            cut_depth (int, optional): 切割深度，即网格的层数，大于1时，精度无效. Defaults to 1.
            const (Constants, optional): 误差控制常量. Defaults to Constants.DEFAULT.
        """
        super().__init__(const)
        self.poly=poly
        self.order=order
        self.covered_points=covered_points
        self.precision=precision
        self.cut_depth=cut_depth
    def _preprocess(self) -> None:
        pass
    def get_result(self) -> list[Loop]:
        """获取最大矩形

        Returns:
            list[Loop]: 第1..order大内接矩形.
        """
        # 1. 根据顶点坐标和精度切分网格，将BoundingBox切割为m*n个cell
        x, y = self._cut_bounds(poly, self.precision, max_depth=self.cut_depth)
        print("网格数=", len(x), "*", len(y))

        # 2. 计算每个cell是否在多边形内
        cell_inside = self._get_01matrix(self.poly, x, y)

        # 3. 计算每个要覆盖的点在哪个cell里，保存在i_covered_points
        idx_covered_region = self._put_covered_points_into_cells(
            x, y, self.order, self.covered_points
        )

        # 4. 计算第k大矩形
        rects = []
        for k in range(self.order):
            iminX, iminY, imaxX, imaxY = self._get_max_submatrix(
                cell_inside, x, y, idx_covered_region[k]
            )
            if imaxX > iminX and imaxY > iminY:
                rects.append(
                    Loop.from_nodes(
                        [
                            Node(x[iminX], y[iminY]),
                            Node(x[imaxX], y[iminY]),
                            Node(x[imaxX], y[imaxY]),
                            Node(x[iminX], y[imaxY]),
                        ]
                    )
                )
                # 计算完后给对应的area区域赋值-const.MAX_NUM
                cell_inside[iminX + 1 : imaxX + 1, iminY + 1 : imaxY + 1] = False
            else:
                rects.append(None)

        return rects
    
    @Timer()
    def _cut_bounds(
        self, poly: Poly, precision: float = -1, max_depth: int = 1
    ) -> tuple[list[float], list[float]]:
        """根据顶点坐标和细分网格，计算xy坐标用于切割BoundingBox

        Args:
            poly (Poly): _description_
            precision (float, optional): _description_. Defaults to -1.
            max_depth (int, optional): _description_. Defaults to 1.

        Returns:
            tuple[list[float], list[float]]: x和y方向的切割点坐标
        """
        edges = poly.edges()
        nodes = poly.nodes()
        int_nodes = self._intersection_nodesXY(edges, nodes)  # 用所有顶点xy切割边
        mbb=poly.exterior.get_mbb()
        int_gridX, int_gridY = self._intersection_gridXY(
            [mbb[0].x,mbb[0].y,mbb[1].x,mbb[1].y], edges, precision
        )  # 用细分网格xy切割边
        more_nodes = []
        new_nodes = int_nodes + int_gridX + int_gridY
        for depth in range(max_depth - 1):
            new_nodes = self._intersection_nodesXY(edges, new_nodes)
            more_nodes += new_nodes
        x = [node.x for node in nodes + int_nodes + int_gridY + more_nodes]
        y = [node.y for node in nodes + int_nodes + int_gridX + more_nodes]
        ListTool.sort_and_overkill(x)
        ListTool.sort_and_overkill(y)
        return x, y
    
    def _intersection_nodesXY(self, edges: list[Edge], nodes: list[Node]) -> list[Node]:
        """用顶点坐标切割边"""
        intersections = []
        for node in nodes:
            nextL, nextR, nextU, nextD = (None,)*4  # 只切割上下左右最近的边
            minL, minR, minU, minD = (self.const.MAX_VAL,)*4
            for edge in edges:
                if (
                    abs(edge.s.x - edge.e.x) < self.const.TOL_DIST
                    or abs(edge.s.y - edge.e.y) < self.const.TOL_DIST
                ):
                    continue  # 横平竖直不用切
                new_node, p = edge.point_at(x=node.x)
                if p is not None and p > 0 + self.const.TOL_VAL and p + self.const.TOL_VAL < 1:
                    if (
                        new_node.y > node.y + self.const.TOL_DIST
                        and new_node.y - node.y < minU
                    ):
                        minU = new_node.y - node.y
                        nextU = new_node
                    if (
                        new_node.y + self.const.TOL_DIST < node.y
                        and node.y - new_node.y < minD
                    ):
                        minD = node.y - new_node.y
                        nextD = new_node
                new_node, p = edge.point_at(y=node.y)
                if p is not None and p > 0 + self.const.TOL_VAL and p + self.const.TOL_VAL < 1:
                    if (
                        new_node.x > node.x + self.const.TOL_DIST
                        and new_node.x - node.x < minR
                    ):
                        minR = new_node.x - node.x
                        nextR = new_node
                    if (
                        new_node.x + self.const.TOL_DIST < node.x
                        and node.x - new_node.x < minL
                    ):
                        minL = node.x - new_node.x
                        nextL = new_node
            if nextL is not None:
                intersections.append(nextL)
            if nextR is not None:
                intersections.append(nextR)
            if nextU is not None:
                intersections.append(nextU)
            if nextD is not None:
                intersections.append(nextD)
        return intersections

    def _intersection_gridXY(
        self, bounds: list[float], edges: list[Edge], precision: float = -1.0
    ) -> tuple[list[Node], list[Node]]:
        """用定距网格切割斜边"""
        if precision > self.const.TOL_VAL:
            ibounds = [math.ceil(bound / precision) for bound in bounds]
        else:
            ibounds = [0] * 4
        x = [i * precision for i in range(ibounds[0], ibounds[2])]
        y = [i * precision for i in range(ibounds[1], ibounds[3])]
        nodesX, nodesY = [], []
        for edge in edges:
            if (
                abs(edge.s.x - edge.e.x) < self.const.TOL_DIST
                or abs(edge.s.y - edge.e.y) < self.const.TOL_DIST
            ):
                continue  # 横平竖直不用切
            for xi in x:
                new_node, p = edge.point_at(x=xi)
                if p is not None and p > 0 + self.const.TOL_VAL and p + self.const.TOL_VAL < 1:
                    nodesX.append(new_node)
            for yi in y:
                new_node, p = edge.point_at(y=yi)
                if p is not None and p > 0 + self.const.TOL_VAL and p + self.const.TOL_VAL < 1:
                    nodesY.append(new_node)
        return nodesX, nodesY

    @Timer()
    def _get_01matrix(self, poly: Poly, x: list[float], y: list[float]) -> np.ndarray:
        """计算每个cell是否在多边形内。cellInside[0,:]=cellInside[:,0]=cellInside[m,:]=cellInside[:,n]=False"""
        poly = Polygon(*poly.offset(dist=-self.const.TOL_DIST).to_array())
        prepare(poly)
        m, n = len(x), len(y)
        ptmat = np.zeros((m, n), dtype=bool)
        for i in range(m):
            for j in range(n):
                ptmat[i, j] = poly.covers(Point(x[i], y[j]))
        mat = np.zeros((m + 1, n + 1), dtype=bool)
        for i in range(1, m):
            for j in range(1, n):
                mat[i, j] = (
                    ptmat[i - 1, j - 1]
                    and ptmat[i, j - 1]
                    and ptmat[i - 1, j]
                    and ptmat[i, j]
                )
        return mat

    @Timer()
    def _put_covered_points_into_cells(
        self,
        x: list[float],
        y: list[float],
        order: int,
        covered_points: list[list[Node]],
    ) -> list[tuple[int, int, int, int]]:
        """计算每组covered points的外包矩形所覆盖的cells。如果Point在边界外，就放到(0,0)里"""
        if len(covered_points) < order:
            covered_points += [[] for _ in range(order - len(covered_points))]
        idx_covered_points = [[] for _ in range(order)]
        for k in range(order):
            if len(covered_points[k]) == 0:
                continue
            mini, minj, maxi, maxj = self.const.MAX_VAL, self.const.MAX_VAL, 0, 0
            for l in range(len(covered_points[k])):
                if (
                    covered_points[k][l].x < x[0]
                    or covered_points[k][l].y < y[0]
                    or covered_points[k][l].x > x[-1]
                    or covered_points[k][l].y > y[-1]
                ):
                    idx_covered_points[k] = (0, 0, 0, 0)
                    break
                flagi, i = ListTool.search_value(x, covered_points[k][l].x)
                flagj, j = ListTool.search_value(y, covered_points[k][l].y)
                mini, minj, maxi, maxj = (
                    min(mini, i),
                    min(minj, j),
                    max(maxi, i),
                    max(maxj, j),
                )
            else:
                idx_covered_points[k] = (mini, minj, maxi, maxj)
        return idx_covered_points

    @Timer()
    def _get_max_submatrix(
        self,
        cell_inside: np.ndarray,
        x: list[float],
        y: list[float],
        idx_covered_region: tuple[int, int, int, int] = None,
    ) -> tuple[int, int, int, int]:
        """返回四个bounds在xy中的index"""
        m, n = len(x), len(y)
        # u、l、r表示从当前格子向上、向左、向右，最多有多少个连续的格子
        u, l, r = (
            np.zeros((m, n + 1), dtype=int),
            np.zeros((m, n + 1), dtype=int),
            np.zeros((m, n + 1), dtype=int),
        )
        for i in range(1, m):
            for j in range(1, n):
                u[i, j] = u[i - 1, j] + 1 if cell_inside[i, j] else 0
                l[i, j] = l[i, j - 1] + 1 if cell_inside[i, j] else 0
            for j in range(n - 1, 0, -1):
                r[i, j] = r[i, j + 1] + 1 if cell_inside[i, j] else 0
        maxs = -self.const.MAX_VAL
        u0, l0, d0, r0 = 0, 0, -1, -1
        # lw、rw表示从当前格子向上u[i,j]个格子到顶，然后再向左、向右最多能扩展多少个格子
        lw, rw = np.zeros((m, n + 1), dtype=int), np.zeros((m, n + 1), dtype=int)
        for i in range(1, m):
            for j in range(1, n):
                if u[i, j] > 1:
                    lw[i, j] = min(lw[i - 1, j], l[i, j])
                    rw[i, j] = min(rw[i - 1, j], r[i, j])
                else:
                    lw[i, j] = l[i, j]
                    rw[i, j] = r[i, j]
                # ut、lt、dt、rt表示此时（以当前格子为底，向上到顶，再左右扩展）的四条边
                ut, lt, dt, rt = i - u[i, j], j - lw[i, j], i, j + rw[i, j] - 1
                s = (x[dt] - x[ut]) * (y[rt] - y[lt])  # 计算矩形面积
                if s > maxs and (
                    (len(idx_covered_region) == 0)
                    or (
                        ut + 1 <= idx_covered_region[0]
                        and lt + 1 <= idx_covered_region[1]
                        and idx_covered_region[2] <= dt
                        and idx_covered_region[3] <= rt
                    )
                ):
                    maxs = s
                    u0, l0, d0, r0 = ut, lt, dt, rt
        return u0, l0, d0, r0
    def _postprocess(self) -> None:
        pass
class MergeLineAlgo(GeomAlgo):
    def __init__(self,lines:list[Edge],preserve_intersections:bool=False,compare:Callable[[Edge,Edge],int]=None,const:Constant=None) -> None:
        """合并重叠的线段

        Args:
            lines (list[Edge]): 待合并的线段.
            preserve_intersection (bool, optional): 是否在交点处打断. Defaults to False.
            compare (Callable[[Edge,Edge],int], optional): 线段的优先级==0(等于)|==1(大于)|==-1(小于); 合并时保留较大的. Defaults to None (==0).
            const (Constants, optional): 误差控制常量. Defaults to Constants.DEFAULT.
        """
        super().__init__(const)
        self.lines=lines
        self.merged_lines:list[Edge]=[] 
        self.preserve_intersections=preserve_intersections
        self.compare=compare or (lambda a,b:0)
    def get_result(self)->list[Edge]:
        """获取合并后的线段

        Returns:
            list[Edge]: 合并后的线段.
        """
        # 1.前处理
        self._preprocess()
        # 2.按平行共线分组
        parallel_groups=self._get_parallel_groups(self.lines)
        collinear_groups=[]
        for group in parallel_groups:
            collinear_groups+=self._get_collinear_groups(group)
        # 3.合并重叠的线段
        for i,group in enumerate(collinear_groups):
            self.merged_lines+=self._merge_collinear_lines(group)
            # self.merged_lines+=self._merge_collinear_lines_by_segtree(group)
        # 4.后处理
        self._postprocess()
        return self.merged_lines
    def _preprocess(self)->None:
        """前处理"""
        Geom.push_const(self.const)
        Domain1d.push_const(self.const)
        Domain1d.push_compare(self.compare)
        # 去除0线段
        self.lines=list(filter(lambda line: not line.is_zero(),self.lines))
        # 角度转换到[0,pi)范围内
        for line in self.lines:
            if (math.pi<=line.angle+self.const.TOL_ANG<math.pi*2):
                line.reverse()
            if line.angle+self.const.TOL_ANG>math.pi*2:
                line.angle=0
    def _postprocess(self)->None:
        """后处理"""
        # 按需打断
        if self.preserve_intersections:
            self.merged_lines=BreakLineAlgo([self.merged_lines],const=self.const).get_result()[0]
        Geom.pop_const()
        Domain1d.pop_const()
        Domain1d.pop_compare()
    def _get_parallel_groups(self,lines:list[Edge]):
        """按平行线分组"""
        parallel_groups=[]
        lines.sort(key=lambda line:line.angle)
        current_angle=-self.const.MAX_VAL
        for line in lines:
            if line.angle-current_angle>self.const.TOL_ANG: # !parallel
                new_group=[line]
                parallel_groups.append(new_group)
                current_angle=line.angle
            else: 
                new_group.append(line)
        return parallel_groups
    def _get_collinear_groups(self,lines:list[Edge]):
        """将平行线按共线分组，组内按起点排序"""
        # 找一根最长的，作为方向向量
        longest_line=max(lines,key=lambda line:line.length)
        # 沿法向量（右转90度）排序
        unit_vector=longest_line.to_vec3d().unit() #单位向量
        normal_vector=unit_vector.cross(Vec3d(0,0,1)) #法向量
        # 分组
        collinear_groups:list[list[Edge]]=[]
        lines.sort(key=lambda line:line.s.to_vec3d().dot(normal_vector))    
        current_dist=-self.const.MAX_VAL
        for line in lines:
            dist_i=line.s.to_vec3d().dot(normal_vector) # 投影
            if dist_i-current_dist>self.const.TOL_DIST: # !collinear
                new_group=[line]
                collinear_groups.append(new_group)
                current_dist=dist_i
            else: 
                new_group.append(line)
        # 组内按起点排序
        for group in collinear_groups:
            group.sort(key=lambda line:line.s.to_vec3d().dot(unit_vector))
        return collinear_groups
    def _merge_collinear_lines(self,unmerged_lines:list[Edge]):
        """顺序合并排好序的共线的线段"""
        lines=unmerged_lines.copy()
        # 找一根最长的，作为方向向量
        longest_line=max(lines,key=lambda line:line.length)
        unit_vector=longest_line.to_vec3d().unit()
        proj=lambda p:p.to_vec3d().dot(unit_vector)
        i=1 # 当前待合并的线段index
        while i<len(lines): 
            # 每次循环将当前的lines[i]线段合并到左侧。当新增/改变了任何线段起点时，需要维护lines有序
            # 意味着i左侧的lines[0..i-1]已完成合并，互相没有交集
            # 并确保当前lines[i]只可能与lines[i-1]有交集，和lines[0..i-2]都没有交集
            l1,r1=proj(lines[i-1].s),proj(lines[i-1].e)
            l2,r2=proj(lines[i].s),proj(lines[i].e)
            if l2>r1+self.const.TOL_DIST: # 1.没有交集，就不合并直接加入；不影响lines有序性
                i+=1
            elif r2<=r1+self.const.TOL_DIST: # 2.完全包含(含端点重合)，先比较优先级
                match self.compare(lines[i],lines[i-1]):
                    case 0|-1: # 2.1.线段i的优先级和i-1相等或者较低，就不参与合并直接被消掉；不影响lines有序性
                        lines.pop(i)
                    case 1: # 2.2.线段i优先级较高，就把线段i-1切三段，把中间一段消掉；需要维护lines有序性
                        line_1=copy(lines[i-1])
                        line_1.s,line_1.e=lines[i-1].s,lines[i].s
                        line_3=copy(lines[i-1])
                        line_3.s,line_3.e=lines[i].e,lines[i-1].e
                        if not line_1.is_zero():  
                            lines[i-1]=line_1
                        else:  # 被切没了的情况
                            lines.pop(i-1) 
                            i-=1
                        if not line_3.is_zero(): # 新增的这段影响lines顺序，插入排序
                            _,pos=ListTool.search_value(lines,proj(line_3.s),key=lambda line:proj(line.s))
                            lines.insert(pos,line_3)
                        i+=1
            elif r2>r1+self.const.TOL_DIST: # 3.相交且需要延长，先比较优先级
                match self.compare(lines[i],lines[i-1]):
                    case 0: # 3.1.线段i的优先级和i-1相等，就直接延长线段i-1，并删掉线段i；不影响lines有序性
                        lines[i-1].s,line[i-1].e=lines[i-1].s,lines[i].e
                        lines.pop(i)
                    case 1: # 3.2.线段i的优先级较高，就切割线段i-1；不影响lines有序性
                        lines[i-1].s,lines[i-1].e=lines[i-1].s,lines[i].s
                        if not lines[i-1].is_zero():
                            i+=1
                        else:  # 被切没了的情况
                            lines.pop(i-1)
                    case -1: # 3.3.线段i的优先级较低，就切割线段i；可能影响lines顺序
                        lines[i].s,lines[i].e=lines[i-1].e,lines[i].e
                        new_pos=i
                        while new_pos<len(lines)-1 and proj(lines[new_pos].s)>proj(lines[new_pos+1].s)+const.TOL_DIST:  # 重新排序
                            lines[new_pos],lines[new_pos+1]=lines[new_pos+1],lines[new_pos]
                            new_pos+=1
                        if new_pos==i: i+=1  # 对顺序没影响的情况
        return lines
    def _merge_collinear_lines_by_segtree(self,unmerged_lines:list[Edge]):
        """用线段树合并共线的线段"""
        # 找一根最长的，作为方向向量
        longest_line=max(unmerged_lines,key=lambda line:line.length)
        unit_vector=longest_line.to_vec3d().unit()
        proj=lambda p:p.to_vec3d().dot(unit_vector)
        domains=[]
        for line in unmerged_lines:
            domains.append(Domain1d(proj(line.s),proj(line.e),line))
            domains[-1].line=line
        seg_tree=SegmentTree(domains,self.const)
        merged_domains=seg_tree.get_leaf_segs()
        merged_lines=[]
        for dom in merged_domains:
            if len(merged_lines)>0 and self.compare(dom.line,merged_lines[-1])==0:
                merged_lines[-1].e=dom.line.e
            elif dom.value>0:
                merged_lines.append(dom.line)
        return merged_lines
class BreakLineAlgo(GeomAlgo):
    def __init__(self,line_groups:list[list[Edge]],const:Constant=None) -> None:
        """线段打断，并保留原先的分组

        Args:
            line_groups (list[list[Edge]]): 若干个分组，每组包含若干条线段.
            const (Constants, optional): 误差控制常量. Defaults to Constants.DEFAULT.
        """
        super().__init__(const=const)
        self.line_groups=line_groups
        self.all_lines:list[Edge]=[]
        self.broken_line_groups:list[list[Edge]]=[]
    def get_result(self)->list[list[Edge]]:
        """获取打断的结果

        Returns
        list[list[Edge]]: 若干个分组，每组包含若干条线段，和打断前的分组一致.
        """
        self._preprocess()
        break_points=self._get_break_points()
        self.broken_lines=self._rebuild_lines(break_points)
        self._postprocess()
        return self.broken_lines
    def _preprocess(self)->None:
        """预处理"""
        super()._preprocess()
        self.all_lines=[line for group in self.line_groups for line in group]
        # 去除0线段
        self.all_lines=list(filter(lambda line: not line.is_zero(),self.all_lines))
    def _postprocess(self)->None:
        """后处理"""
        super()._postprocess()
    def _get_break_points(self)->dict[Edge:list[Node]]:
        """获取线段上的断点，没有排序，也没有去重"""
        visited={line:set() for line in self.all_lines} # 记录已经求过交点的线段
        break_points={line:[line.s,line.e] for line in self.all_lines} # 记录线段上的断点
        rt=STRTree(self.all_lines)
        for line in self.all_lines:
            neighbors=rt.query_idx(line.get_mbb(),tol=1.0)
            for other_idx in neighbors:
                other=self.all_lines[other_idx]
                if other in visited[line]: continue # 这俩已经求过了，就不再算了
                visited[line].add(other)
                visited[other].add(line)
                # 共线且有重叠的情况
                if line.is_collinear(other):
                    overlap=line.overlap(other)
                    for edge in overlap:
                        if edge.is_zero(): continue
                        break_points[line].extend([edge.s,edge.e])
                        break_points[other].extend([edge.s,edge.e])
                # 相交的情况
                if line.intersects(other):
                    intersection=line.intersection(other)
                    break_points[line].extend(intersection)
                    break_points[other].extend(intersection)
        return break_points
    def _rebuild_lines(self,break_points:dict[Edge:list[Node]])->list[list[Edge]]:
        """根据断点重构线段"""
        broken_lines=[]
        for group in self.line_groups:
            new_group=[]
            for line in group:
                break_points[line].sort(key=lambda p:line.get_point_param(p))
                pre=break_points[line][0]
                for p in break_points[line]:
                    if p.equals(pre): continue
                    new_group.append(line.slice_between_points(pre, p))
                    pre=p
            broken_lines.append(new_group)
        return broken_lines
class FindConnectedGraphAlgo(GeomAlgo):
    def __init__(self,lines:list[Edge],const:Constant=None) -> None:
        """求连通图

        Args:
            edges (list[Edge]): 所有线段，需要打断.
            const (Constants, optional): 误差控制常量. Defaults to Constants.DEFAULT.
        """
        super().__init__(const)
        self.lines=lines
        self.connected_graphs:list[list[Edge]]=[]
    def get_result(self)->list[list[Edge]]:
        """获取结果

        Returns
        -------
        list[list[Edge]]
            n个连通图
        """
        self._preprocess()
        self.connected_graphs=self._find_connected_graphs()
        self._postprocess()
        return self.connected_graphs
    def _find_connected_graphs(self)->list[list[Edge]]:
        """广度优先搜索"""
        if len(self.lines)==0: return []
        res=[]
        rt=STRTree(self.lines)
        q,head=[self.lines[0]],0 #一个连通分量
        visited_lines={self.lines[0]}
        pre_head=0 #保证lines[0..pre_head]是已访问的线段
        while True:
            if head==len(q):
                res.append(q)
                while pre_head<len(self.lines) and self.lines[pre_head] in visited_lines:
                    pre_head+=1 
                if pre_head<len(self.lines): 
                    q=[self.lines[pre_head]]
                    head=0
                else: break
            current_line=q[head]
            head+=1
            for node in [current_line.s,current_line.e]:
                neighbor_lines=rt.query(node.get_mbb(),tol=self.const.TOL_DIST*2)
                for line in neighbor_lines:
                    if line not in visited_lines:
                        if node.dist(line.s)<self.const.TOL_DIST or node.dist(line.e)<self.const.TOL_DIST:
                            q.append(line)
                            visited_lines.add(line)
        return res
    def _preprocess(self)->None:
        """预处理"""
        # 去除0线段
        self.lines=list(filter(lambda line: not line.is_zero(),self.lines))
    def _postprocess(self)->None:
        """后处理"""
        pass
class FindOutlineAlgo(GeomAlgo):
    def __init__(self,edges:list[Edge],const:Constant=None) -> None:
        """求单个连通图形的外轮廓

        Args:
            edges (list[Edge]): 所有线段，无需打断.
            const (Constants, optional): 误差控制常量. Defaults to Constants.DEFAULT.
        """
        super().__init__(const)
        self.edges=edges
    def get_result(self)->Loop:
        """获取结果（逆时针）"""
        self._preprocess()
        start_edge=self._find_start_edge()
        rt_edges=STRTree(self.edges)
        outline=self._find_outline(rt_edges,start_edge)
        return outline
    def _preprocess(self)->None:
        """前处理"""
        # 去除0线段
        self.edges=list(filter(lambda line: not line.is_zero(),self.edges))
    def _find_start_edge(self)->Edge: 
        """找起始边：先找x最小的点，然后向左出发"""
        start_edge=min(self.edges,key=lambda edge:edge.get_mbb()[0].x)
        mbb=start_edge.get_mbb()
        left_bound=LineSeg(mbb[0],Node(mbb[0].x,mbb[1].y))
        intersections=start_edge.intersection(left_bound)
        if len(intersections)==1:
            start_node=intersections[0]
        elif len(intersections)>1:
            start_node=min(intersections,key=lambda p:p.x)
        return LineSeg(Node(start_node.x+self.const.TOL_DIST*2,start_node.y),start_node)
    def _find_outline(self,rt_edges:STRTree ,start_edge:Edge)->Loop: 
        """顺着start_edge逆时针找一圈外轮廓"""
        outline:list[Edge]=[]
        pre_edge=start_edge
        this_node=pre_edge.e
        while True: # 每次循环从pre_edge出发，找下一条边，直到回到起点
            # 搜索与当前出发点临近的边
            nearest_edges_i=rt_edges.query_idx(this_node.get_mbb(),tol=self.const.TOL_DIST) 
            # 求当前顶点到这些边的端点的连线的集合
            edges_from_this_node:list[Edge] =[]
            for i in nearest_edges_i:
                edge=self.edges[i]
                if not this_node.is_on_edge(edge): continue
                if not this_node.equals(edge.s):
                    edges_from_this_node.append(edge.opposite().slice_between_points(this_node,edge.s))
                if not this_node.equals(edge.e):
                    edges_from_this_node.append(edge.slice_between_points(this_node,edge.e))
            # 从pre_edge.opposite出发，对这些连线按角度[0,2pi)逆时针排序
            op=pre_edge.opposite()
            edges_from_this_node.sort(key=lambda edge:op.tangent_at(0).angle_to(edge.tangent_at(0)))
            # 从连线集合中找到pre_edge的下一个角度
            i=0
            while (i<len(edges_from_this_node) 
                   and (op.tangent_at(0).angle_to(edges_from_this_node[i].tangent_at(0))<self.const.TOL_ANG 
                        or op.tangent_at(0).angle_to(edges_from_this_node[i].tangent_at(0))>2*math.pi-self.const.TOL_ANG)):
                i+=1
            new_edge=edges_from_this_node[i%len(edges_from_this_node)] # 如果走到死路了（i==l），就倒回去
            # 求所有与new_edge可能相交的线
            nearest_edges_i=rt_edges.query_idx(new_edge.get_mbb(),tol=self.const.TOL_DIST)
            # 遍历相交的线，取距离最近的一个交点(起点除外)，作为下一个顶点
            min_dist=new_edge.length
            next_node=new_edge.e
            for i in nearest_edges_i:
                nearest_edge=self.edges[i]
                intersections=new_edge.intersection(nearest_edge)
                for p in intersections:
                    d=this_node.dist(p)
                    if not p.equals(this_node) and d<min_dist and p.is_on_edge(new_edge): # 不是起点，且距离最近
                        min_dist=d
                        next_node=p
            new_edge=Edge(this_node,next_node)
            if len(outline)>0 and new_edge.s.equals(outline[0].s) and new_edge.e.equals(outline[0].e): # 回到起点就结束
                break
            else: # 保存当前边，继续找下一条边
                outline.append(new_edge)
                pre_edge=new_edge
                this_node=next_node            
        return Loop(outline)
    def _postprocess(self) -> None:
        pass
class FindLoopAlgo(GeomAlgo):  #TODO
    def __init__(self,edges:list[Edge],const:Constant=None) -> None:
        """搜索线段构成的所有封闭区域

        Args:
            edges (list[Edge]): 所有线段，无需打断
            const (Constant, optional): 误差控制常量. Defaults to None.
        """
        super().__init__(const=const)
        self.edges:list[Edge]=edges
        self.loops:list[Loop]=[]
    def _preprocess(self)->None:
        super()._preprocess()
        self.edges=BreakLineAlgo(self.edges,self.const).get_result()

    def get_result(self):
        self._preprocess()
        self.loops=self._find_loop()
        self._postprocess()
    def _postprocess(self)->None:
        super()._postprocess()
    def _find_loop(self)->list[Loop]:
        loops:list[Loop] =[]
        visited_edges=set()
        edge_num=sum([len(node.edge_out) for node in self.nodes]) #算总边数
        while edge_num>0: #每次循环找一个环，直到所有边都被遍历过
            new_loop:list[Node]=[]
            for node in self.nodes: #先随便找一条边作为pre_edge
                if len(node.edge_out)>0:
                    pre_edge=node.edge_out[0]
                    break
            while True: #以pre_edge.e为当前结点开始找一个环
                node=pre_edge.e #当前结点
                op=pre_edge.opposite()
                current_angle=op.tangent_at(0).angle #入边的角度
                current_curvature=pre_edge.opposite().curvature_at(0)
                i=len(node.edge_out)-1
                while i>=0 and (node.edge_out[i].tangent_at(0).angle+const.TOL_ANG>=current_angle 
                                or (node.edge_out[i].tangent_at(0).angle-current_angle)<const.TOL_ANG): #按角度找下一条出边 ##########################
                    i-=1
                if node.edge_out[i] in visited_edges:  #如果找到了已访问的边就封闭这个环
                    loops.append(Loop(new_loop)) #先将此环加入list
                    for i in range(len(new_loop)): #并把环上的边都从邻接表里删掉
                        new_loop[i].s.edge_out.remove(new_loop[i])
                    edge_num-=len(new_loop) #从总边数中减去环的边数
                    break
                else: #如果找到的不是已访问的边
                    new_loop.append(node.edge_out[i]) #就将此边加入环
                    visited_edges.add(node.edge_out[i]) #标记为已访问
                    pre_edge=node.edge_out[i] #接着找下一条边
        return loops
class FindRoomAlgo(GeomAlgo): #TODO
    def __init__(self,edges:list[Wall],const:Constant=None) -> None:
        self.const=const or Constant.default()
        self.edges:list[Edge]=edges
        self.loops:list[Loop]=[]
    def _preprocess(self) -> None:
        super()._preprocess()
    def _postprocess(self) -> None:
        super()._postprocess()
    def make_cover_tree(loops:list[Loop])->list[TreeNode]:
        loops.sort(key=lambda loop:abs(loop.area),reverse=True)  # 按面积排序，确保循环的时候每个TreeNode都有正确的parent
        t:list[TreeNode] =[TreeNode(loop) for loop in loops] #把loop都变成TreeNode
        for i in range(len(t)-1):
            for j in range(i+1,len(t)):
                ni,nj=t[i],t[j]
                ci=ni.obj.covers(nj.obj)
                cj=nj.obj.covers(ni.obj)
                if not ci and not cj: # 没有覆盖关系时，跳过
                    continue
                elif ci and cj:  # 互相覆盖(重合)的时候，取ci.parent的相反方向的环作为外环
                    if ni.parent is None or ni.parent.obj.area>0:  # ni.parent是内环，就让负的覆盖正的，保证正-负-正的关系
                        if ni.obj.area>0:
                            ni,nj=nj,ni
                    else:  # ni.parent是外环，就让正的覆盖负的，保证负-正-负的关系
                        if ni.obj.area<0:
                            ni,nj=nj,ni
                elif cj and not ci:  # j覆盖i且i不覆盖j（ij不重合）时，ij互换 
                    ni,nj=nj,ni
                # 此时确保i覆盖j
                nj.parent=ni
        for i in t:
            if i.parent is not None:
                i.parent.child.append(i)
        return t
class SplitIntersectedLoopsAlgo(GeomAlgo):
    """
    多个相交/自相交环的合并算法。
    方法一（当前采用）：在所有交点处交换边的方向。
    方法二：Winding number algorithm. https://mcmains.me.berkeley.edu/pubs/DAC05OffsetPolygon.pdf
    方法三：Vatti clipping algorithm. https://github.com/dpuyda/triclipper/blob/master/docs/how_it_works.md
    """
    def __init__(self,loops:list[Loop],positive:bool=True,ensure_valid:bool=True,const:Constant=None) -> None:
        """合并多个相交的环

        Args:
            loops (list[Loop]): 待合并的环
            positive (bool, optional): 上层是否为正环. Defaults to True.
            ensure_valid (bool, optional): 结果是否为正环. Defaults to True.
            const (Constant, optional): 误差控制常量. Defaults to None.
        """
        super().__init__(const=const)
        self.loops=loops
        self.positive=positive
        self.ensure_valid=ensure_valid
        self.split_loops:list[Loop]=[]
    """
    def _get_overlapping_edges_in_loop(self,loop:Loop)->list[Edge]:
        res=set()
        for i in range(len(loop.edges)-1):
            for j in range(i+1,len(loop.edges)):
                overlap=loop.edges[i].overlap(loop.edges[j])
                if len(overlap)>0 and not overlap[0].is_zero():
                    res.add(loop.edges[i])
                    res.add(loop.edges[j])
        return list(res)
    """
    def _preprocess(self)->None:
        super()._preprocess()
        # 所有边集合
        self.all_edges=[edge for loop in self.loops for edge in loop.edges]
        # 初始化loop顶点的next_edge和dual_node
        for loop in self.loops:
            for i in range(len(loop.edges)):
                loop.edges[i].s.next_edge=loop.edges[i]
                loop.edges[i].s.dual_node_on_next_edge=loop.edges[i-1].e
                loop.edges[i-1].e.next_edge=loop.edges[i]
                loop.edges[i-1].e.dual_node_on_next_edge=loop.edges[i].s
    def get_result(self):
        self._preprocess()
        # 先在所有的交点处打断，并交换方向
        breakpoints=self._get_breakpoints()  # 每条边上的断点及其后继边
        # 然后顺着各点找所有闭合的环
        self.split_loops=self._find_loops(breakpoints,positive=self.positive,ensure_valid=self.ensure_valid)
        # （按需要）去除方向不太对劲的环
        if self.ensure_valid:
            self.split_loops=self._remove_invalid_loops(self.split_loops)
        self._postprocess()
        return self.split_loops
    def _postprocess(self)->None:
        # 当前算法存在永远处理不了的情况：内外同向相切的环。此时稍微offset一下由相切转为相交再处理。（当前是否需要待验证 TODO）
        if (len(self.loops)==len(self.split_loops)==1 and 
                abs(self.loops[0].area-self.split_loops[0].area)<self.loops[0].length*self.const.TOL_DIST+self.const.TOL_AREA):
            self.split_loops=self.split_loops[0].offset(dist=self.const.TOL_DIST,split=False)
        # 经过一次交换处理，自相交的部分可能从一个环上转移到另一个环上。所以需要看看还有没有残留的自相交环，如果有的话需要继续递归处理
        res=[]
        for loop in self.split_loops:
            if loop.has_self_intersection():
                new_loop=SplitIntersectedLoopsAlgo([loop],self.positive,self.ensure_valid,const=self.const).get_result()
                res.extend(new_loop)
            else: res.append(loop)
        self.split_loops=res
        self.split_loops=list(filter(lambda loop:abs(loop.area*2/loop.length)>1,self.split_loops)) # 移除过细的环
        super()._postprocess()
    def _get_breakpoints(self)->dict[Edge:list[Node]]:
        """在所有的交点处打断，并交换方向"""
        breakpoints={edge:[] for edge in self.all_edges}  # 记录每条边上的断点
        # 枚举不相邻的两条边ei、ej
        for i in range(len(self.all_edges)-1):
            ei=self.all_edges[i]
            for j in range(i+1,len(self.all_edges)):
                ej=self.all_edges[j]
                all_pi:list[Node]=[]  # 位于ei上的断点（圆弧相交可能不止一个）
                if ei.intersects(ej):  # 如果相交，就在交点处打断
                    new_breakpoints=ei.intersection(ej)
                    for pi in new_breakpoints:
                        # 求交的时候每条线段的有效范围是[0,1)->[s,e)；只算头，不算尾巴
                        if pi.equals(ei.e) or pi.equals(ej.e): continue
                        all_pi.append(pi)
                elif len(ei.overlap(ej))>0: # 如果重叠且反向，就在端点处打断；只算头不算尾巴
                    if ei.is_on_same_direction(ej): continue
                    if ei.s.is_on_edge(ej) and not ei.s.equals(ej.e):
                        all_pi.append(copy(ei.s))
                    if ej.s.is_on_edge(ei) and not ej.s.equals(ei.e):
                        if not ej.s.equals(ei.s):  # 如果两个头重叠只算一次
                            all_pi.append(copy(ej.s))
                # 在交点处交换方向
                for pi in all_pi:
                    pj=Node(pi.x,pi.y)  # 位于ej上的断点
                    # 交换方向
                    pi.next_edge=self.all_edges[j]
                    pj.next_edge=self.all_edges[i]
                    # 互为对偶
                    pi.dual_node_on_next_edge=pj
                    pj.dual_node_on_next_edge=pi
                    # 记录断点
                    breakpoints[ei].append(pi)
                    breakpoints[ej].append(pj)
        # 每条边按顺序重排断点
        for edge in self.all_edges:
            breakpoints[edge].sort(key=lambda node:edge.get_point_param(node))
            # 加入首尾顶点
            breakpoints[edge].insert(0,edge.s)
            breakpoints[edge].append(edge.e)
        return breakpoints
    def _find_loops(self,breakpoints:dict[Edge:list[Node]],positive:bool,ensure_valid:bool) -> list[Loop]:
        """找所有闭合环"""
        visited_nodes=set()
        split_loops=[]
        for some_edge in self.all_edges:
            for some_point in breakpoints[some_edge]:
                # 先随便从一个没被访问过的顶点开始
                if some_point in visited_nodes: continue
                this_node=some_point
                new_loop_edges:list[Edge]=[]
                while True:  # 以this_node为当前顶点开始找一个环
                    next_edge=this_node.next_edge  # 从this_node出发，应该沿着next_edge这条边走
                    i=breakpoints[next_edge].index(this_node.dual_node_on_next_edge)  # this_node在被打断的next_edge上的位置，即其对偶点的index
                    next_node=breakpoints[next_edge][i+1]  # 切换到next_edge上，下一点作为新的顶点
                    if next_node in visited_nodes:  # 如果找到了已访问的顶点就封闭这个环
                        new_loop=Loop(new_loop_edges)
                        if (not ensure_valid or (abs(new_loop.area)>self.const.TOL_AREA) and (new_loop.area>self.const.TOL_AREA or not positive)):  # 如果非0，且面积为正或原本就是一个负环
                            new_loop.simplify(cull_insig=True)
                            if len(new_loop.edges)>1:
                                split_loops.append(new_loop)  # 将环加入list
                        break
                    else:  # 如果找到的不是已访问的顶点
                        edge_slice=next_edge.slice_between_points(this_node,next_node)
                        if edge_slice is not None: 
                            new_loop_edges.append(edge_slice)  # 就将此边加入环
                        visited_nodes.add(next_node)  # 标记为已访问
                        this_node=next_node  # 接着找下一条边
        return split_loops
    def _remove_invalid_loops(self,split_loops:list[Loop])->list[Loop]:
        """去除被覆盖的同向环"""
        valid_loops:list[Loop] =[]
        for l1 in split_loops:
            for l2 in split_loops:
                if (l1 is l2) or (l1.area>0)!=(l2.area>0) :continue
                if l2.covers(l1): break
            else: valid_loops.append(l1)
        return valid_loops
class MergeWallAlgo(GeomAlgo):
    def __init__(self,walls:list[Wall],const:Constant=None) -> None:
        """合并平行且有重叠的墙

        Args:
            walls (list[Wall]): 任意一组待合并的墙.
            const (Constant, optional): 误差控制常量. Defaults to None.
        """
        super().__init__(const=const)
        self.walls=walls
    def _preprocess(self)->None:
        super()._postprocess()
    def _get_parallel_groups(self,walls:list[Wall])->list[list[Wall]]:
        """按平行分组"""
        parallel_groups=[]
        # 圆弧墙分组
        arc_walls=filter(lambda wall:isinstance(wall.base,Arc),walls)
        arc_groups=groupby(arc_walls,key=lambda wall:wall.base.center)
        parallel_groups.extend()
        # 直墙分组
        line_walls=filter(lambda wall:isinstance(wall.base,LineSeg),walls)
        walls.sort(key=lambda wall:wall.base.angle)
        current_angle=-self.const.MAX_VAL
        for line in lines:
            if line.angle-current_angle>self.const.TOL_ANG: # !parallel
                new_group=[line]
                parallel_groups.append(new_group)
                current_angle=line.angle
            else: 
                new_group.append(line)
        return parallel_groups
    def get_result(self):
        for i in range(len(self.walls)-1):
            for j in range(i+1,len(self.walls)):
                if self.walls[i].base
    def _postprocess(self)->None:
        super()._postprocess()

def _draw_polygon(poly: Polygon | Poly, show:bool=False, *args, **kwargs):
    x, y = poly.exterior.xy
    plt.plot(x, y, *args, **kwargs)
    for hole in poly.interiors:
        x, y = hole.xy
        plt.plot(x, y, *args, **kwargs)
    if show:
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
def _draw_loops(loops:list[Loop],show_node:bool=False,show_text:bool=False,show:bool=False)->None:
    for i,loop in enumerate(loops):
        # if abs(loop.area)<const.TOL_AREA: continue
        color=colors[i % len(colors)]
        line_style="solid" if loop.area>0 else "dashed"
        for j,edge in enumerate(loop.edges):
            if isinstance(edge,LineSeg):
                plt.plot(*edge.to_array().T,color=color,linestyle=line_style)
            elif isinstance(edge,Arc):
                sub_edges=edge.fit()
                for sub_edge in sub_edges:
                    plt.plot(sub_edge.to_array()[:,0], sub_edge.to_array()[:,1],color=color,linestyle=line_style)
            if show_node:
                plt.scatter(edge.s.x,edge.s.y)
                if show_text:
                    plt.text(edge.s.x+1.0*j,edge.s.y+1.0*j,f"{i}.{j}",color="b")
                    plt.plot([edge.s.x,edge.s.x+1.0*j],[edge.s.y,edge.s.y+1.0*j],alpha=0.1)
    if show:
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
def _draw_edges(edges:list[Edge],show_node_text:bool=False,show:bool=False)->None:
    color=colors[i % len(colors)]
    for j,edge in enumerate(edges):
        if isinstance(edge,LineSeg):
            plt.plot(*edge.to_array().T,color=color)
        elif isinstance(edge,Arc):
            sub_edges=edge.fit()
            for sub_edge in sub_edges:
                plt.plot(sub_edge.to_array()[:,0], sub_edge.to_array()[:,1],color=color)
        if show_node_text:
            plt.scatter(edge.s.x,edge.s.y)
            plt.text(edge.s.x+1.0*j,edge.s.y+1.0*j,j,color="b")
            plt.plot([edge.s.x,edge.s.x+1.0*j],[edge.s.y,edge.s.y+1.0*j],alpha=0.1)
            if j==len(edges)-1:
                plt.scatter(edge.e.x,edge.e.y)
                plt.text(edge.e.x+1.0*j+1,edge.e.y+1.0*j+1,j+1,color="b")
                plt.plot([edge.e.x,edge.e.x+1.0*j],[edge.e.y,edge.e.y+1.0*j],alpha=0.1)
    if show:
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()
# %% 最大矩形测试
if 0 and __name__ == "__main__":
    from random import random, seed
    from matplotlib import pyplot as plt

    const=Constant.default()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect(1)

    # 随机多边形
    

    seed(4)
    SCALE = 100000.0
    SHELL_BOX = 50
    HOLE_BOX = 5
    MAX_ROTATION = 90
    while True:
        rand_poly = box(0, 0, 0, 0)
        for i in range(SHELL_BOX):  # SHELL
            new_box = rotate(
                box(
                    random() * SCALE,
                    random() * SCALE,
                    random() * SCALE,
                    random() * SCALE,
                ),
                random() * MAX_ROTATION,
            )
            rand_poly = rand_poly.union(new_box)
        for i in range(HOLE_BOX):  # HOLES
            new_box = rotate(
                box(
                    random() * SCALE,
                    random() * SCALE,
                    random() * SCALE,
                    random() * SCALE,
                ),
                random() * MAX_ROTATION,
            )
            rand_poly = rand_poly.difference(new_box)
        if isinstance(rand_poly, Polygon):
            rand_poly = rand_poly.simplify(tolerance=const.TOL_DIST)
            break
    exterior = Loop.from_nodes([Node(x, y) for x, y in rand_poly.exterior.coords])
    interiors = [
        Loop.from_nodes([Node(x, y) for x, y in hole.coords])
        for hole in rand_poly.interiors
    ]
    poly = Poly(exterior, interiors)

    # 包含点
    ORDER = 1
    POINT_NUM = 2
    # covered_points = [[Node(random() * SCALE, random() * SCALE) for i in range(POINT_NUM)] for j in range(ORDER)]
    covered_points = [[Node(60000, 10000), Node(50000, 15000)]]

    # 求最大矩形
    PRECISION = -1
    CUT_DEPTH = 1

    ins = MaxRectAlgo(
        poly=poly,
        order=ORDER,
        covered_points=covered_points,
        precision=PRECISION,
        cut_depth=CUT_DEPTH,
    )
    rects = ins.get_result()

    _draw_polygon(rand_poly, color="r")
    for i in range(len(rects)):
        if rects[i] is None:
            continue
        # print(i,rects[i].area)
        x, y = rects[i].xy
        plt.fill(x, y, color="b", alpha=0.8 - (0.8 / ORDER) * i)
    for k in range(len(covered_points)):
        for l in range(len(covered_points[k])):
            plt.scatter(
                covered_points[k][l].x,
                covered_points[k][l].y,
                color="r",
                alpha=0.8 - (0.8 / ORDER) * k,
            )
    plt.show()

# %% 线段合并测试
if 0 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    const=Constant.default()
    with open("test/line_set/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    lines=[]
    for obj in j_obj:
        if obj["object_name"]!="line":continue
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(Edge(s,e))

    print(f"{len(lines)} lines before")
    merged_lines=MergeLineAlgo(lines,preserve_intersections=False).get_result()
    print(f"{len(merged_lines)} lines after")
    
    # for line in merged_lines:
    #     plt.plot([line.s.x,line.e.x],[line.s.y,line.e.y])
    # ax = plt.gca()
    # ax.set_aspect(1)
    # plt.show()

# %% 线段打断测试
if 0 and __name__ == "__main__":
    import json
    const=Constant.default()
    with open("./test/line_set/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    print(f"{len(j_obj)} lines before")
    lines=[]
    for obj in j_obj:
        x1,y1,_=obj["start_point"]
        x2,y2,_=obj["end_point"]
        s,e=Node(x1,y1),Node(x2,y2)        
        if s.equals(e):continue
        lines.append(Edge(s,e))
    broken_lines=BreakLineAlgo([lines]).get_result()[0]
    print(f"{len(broken_lines)} lines after")

# %% 外轮廓测试
if 0 and __name__ == "__main__":
    import json
    const=Constant.default()
    with open("./test/find_outline/case_3.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        x1,y1,_=ent["start_point"]
        x2,y2,_=ent["end_point"]
        s=Node(x1,y1)
        e=Node(x2,y2)
        if s.equals(e):continue
        edges.append(Edge(s,e))
    outline=FindOutlineAlgo(edges).get_result()
    print(len(outline.edges),outline.area)
    plt.plot(*outline.xy)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

# %% 连通图测试
if 0 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    const=Constant.default()
    with open("./test/find_wall/case_13.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        if ent["object_name"]=="line" and ent["layer"]=="WALL":
            x1,y1,z1=ent["start_point"]
            x2,y2,z2=ent["end_point"]
            s=Node(x1,y1)
            e=Node(x2,y2)
            if s.equals(e):continue
            edges.append(Edge(s,e))
    edges=BreakLineAlgo([edges]).get_result()[0]
    con_graph=FindConnectedGraphAlgo(edges).get_result()
    print(len(con_graph))
    colors=list(TABLEAU_COLORS)
    for idx,g in enumerate(con_graph):
        color=colors[idx % len(colors)]
        for line in g:
            plt.plot(*line.to_array().T,color=color)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

# %% 连通图+外轮廓测试
if 0 and __name__ == "__main__":
    import json
    const=Constant.default()
    with open("./test/find_wall/case_1.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    edges:list[Edge]=[]
    for ent in j_obj:
        if ent["object_name"]=="line" and ent["layer"]=="WALL":
            x1,y1,z1=ent["start_point"]
            x2,y2,z2=ent["end_point"]
            s=Node(x1,y1)
            e=Node(x2,y2)
            if s.equals(e):continue
            edges.append(Edge(s,e))
    edges=BreakLineAlgo([edges]).get_result()[0]
    con_graph=FindConnectedGraphAlgo(edges).get_result()
    outlines=[FindOutlineAlgo(edges).get_result() for edges in con_graph]
    print(len(outlines))
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    colors=list(TABLEAU_COLORS)
    for idx,g in enumerate(outlines):
        color=colors[idx % len(colors)]
        for line in g.edges:
            plt.plot(*line.to_array().T,color=color)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

# %% 线段合并测试，带优先级比较
if 0 and __name__ == "__main__":
    import json,random
    import matplotlib.pyplot as plt
    const=Constant.default()

    # with open("./test/merge_line/case_1.json",'r',encoding="utf8") as f:
    #     j_obj=json.load(f)
    # edges:list[Edge]=[]
    # for ent in j_obj:
    #     if ent["object_name"]=="line" and ent["layer"]=="WALL":
    #         x1,y1,z1=ent["start_point"]
    #         x2,y2,z2=ent["end_point"]
    #         s=Node(x1,y1)
    #         e=Node(x2,y2)
    #         if s.equals(e):continue
    #         edges.append(Edge(s,e))

    lines:list[LineSeg]=[]
    limits=(0,10000,1000)
    random.seed(0)
    for i in range(10):
        s=Node(random.random()*(limits[1]-limits[0])+limits[0],0)
        # e=Node(random.random()*(limits[1]-limits[0])+limits[0],0)
        e=Node(s.x+1000,0)
        lw=random.random()*limits[2]
        # s=Node(random.randint(limits[0],limits[1]),0)
        # e=Node(random.randint(limits[0],limits[1]),0)
        # lw=random.randint(0,limits[2])
        lines.append(LineSeg(s,e))
        lines[-1].lw,lines[-1].rw=lw,0

    plt.subplot(2,1,1)
    for i,line in enumerate(lines):
        plt.plot([line.s.x,line.e.x],[line.s.y+line.lw+line.rw,line.e.y+line.lw+line.rw])

    print(f"{len(lines)} lines before")
    def compare(self,a:Edge,b:Edge): 
        if a is None: return -1
        if b is None: return 1
        if abs(a.lw+a.rw-(b.lw+b.rw))<const.TOL_DIST: return 0
        elif a.lw+a.rw>b.lw+b.rw: return 1
        else: return -1
    merged_lines=MergeLineAlgo(lines,preserve_intersections=False,compare=compare).get_result()
    print(f"{len(merged_lines)} lines after")

    plt.subplot(2,1,2)
    for i,line in enumerate(merged_lines):
        plt.plot([line.s.x,line.e.x],[line.s.y+line.lw+line.rw,line.e.y+line.lw+line.rw])

    plt.show()

    # CASE_ID="6"

    # with open(f"./test/merge_line/case_{CASE_ID}.json",'w',encoding="utf8") as f:
    #     json.dump(lines,f,ensure_ascii=False,default=lambda x:x.__dict__)
    # with open(f"./test/merge_line/case_{CASE_ID}_out.json",'w',encoding="utf8") as f:
    #     json.dump(merged_lines,f,ensure_ascii=False,default=lambda x:x.__dict__)
# %% 合并相交环测试
if 1 and __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    from matplotlib.colors import TABLEAU_COLORS
    from tool.dwg_converter.json_parser import polyline_to_loop
    colors=list(TABLEAU_COLORS)
    const=Constant.default()
    # const=Constant("split_loop",tol_area=1e3,tol_dist=1e-2)

    CASE_ID = "12.2"  ################ TEST #################

    with open(f"test/split_loop/case_{CASE_ID}.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    loops=polyline_to_loop(j_obj)

    with Timer(tag="split_loop"):
        split_loops=SplitIntersectedLoopsAlgo(loops,False,False,const=const).get_result()
    split_loops.sort(key=lambda loop:loop.area)

    print(len(split_loops))
    _draw_loops(split_loops,show_node=False,show_text=False,show=True)

    # 输出标准结果
    # with open(f"test\split_loop\case_{CASE_ID}_out.json",'w',encoding="utf8") as f:
    #     json.dump([loop.area for loop in split_loops],f,ensure_ascii=False)
