from abc import ABC, abstractmethod
import math
import numpy as np
from typing import Callable
from queue import PriorityQueue

from lib.linalg import Vec3d
from lib.interval import Interval1d
from lib.geom import (
    Node,Edge,Line,LineSeg,Arc,Circle,Loop,Polygon,
    GeomUtil,
    )
from lib.utils import Timer,Constant as Const,ListTool,ComparerInjector
from lib.index import STRTree,SegmentTree, TreeNode

from lib.geom_plotter import CADPlotter as plt
import os
if os.environ.get("DEBUG",False):
    print("debugging")

class GeomAlgo(ABC):
    def __init__(self) -> None: ...
    @abstractmethod
    def _preprocess(self)->None: ...
    @abstractmethod
    def get_result(self): ...
    @abstractmethod
    def _postprocess(self)->None: ...

class MaxRectAlgo(GeomAlgo):  # TODO
    """多边形与坐标轴平行的最大内接矩形.

    Args:
        poly (Poly): 多边形.
        order (int, optional): 最大矩形的数量. Defaults to 1.
        covered_points (list[list[Node]], optional): 第1..order大矩形必须包含的点坐标，没有则对应位置=None. Defaults to None.
        precision (float, optional): 斜线的xy分割网格尺寸，单位mm；-1.0即不分割. Defaults to -1.0.
        cut_depth (int, optional): 切割深度，即网格的层数，大于1时，精度无效. Defaults to 1.
        const (Const, optional): 误差控制常量. Defaults to Const.DEFAULT.
    
    Todo:
        * 适配新的geom接口
        * 圆弧边界
    """    
    def __init__(
            self,
            poly: Polygon,
            order: int = 1,
            covered_points: list[list[Node]] = None,
            precision: float = -1.0,
            cut_depth=1,
            const:Const=None,
        ) -> None:
        super().__init__()
        self.poly=poly
        self.order=order
        self.covered_points=covered_points
        self.precision=precision
        self.cut_depth=cut_depth
    def _preprocess(self) -> None: ...
    def get_result(self) -> list[Loop]:
        """获取最大矩形

        Returns:
            list[Loop]: 第1..order大内接矩形.
        """
        # 1. 根据顶点坐标和精度切分网格，将BoundingBox切割为m*n个cell
        x, y = self._cut_bounds(self.poly, self.precision, max_depth=self.cut_depth)
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
    
    @Timer
    def _cut_bounds(
        self, poly: Polygon, precision: float = -1, max_depth: int = 1
    ) -> tuple[list[float], list[float]]:
        """根据顶点坐标和细分网格，计算xy坐标用于切割BoundingBox

        Args:
            poly (Poly): _description_
            precision (float, optional): _description_. Defaults to -1.
            max_depth (int, optional): _description_. Defaults to 1.

        Returns:
            tuple[list[float], list[float]]: x和y方向的切割点坐标
        """
        edges = poly.edges
        nodes = poly.nodes
        int_nodes = self._intersection_nodesXY(edges, nodes)  # 用所有顶点xy切割边
        aabb=poly.shell.get_aabb()
        int_gridX, int_gridY = self._intersection_gridXY(
            [aabb[0].x,aabb[0].y,aabb[1].x,aabb[1].y], edges, precision
        )  # 用细分网格xy切割边
        more_nodes = []
        new_nodes = int_nodes + int_gridX + int_gridY
        for depth in range(max_depth - 1):
            new_nodes = self._intersection_nodesXY(edges, new_nodes)
            more_nodes += new_nodes
        x = [node.x for node in nodes + int_nodes + int_gridY + more_nodes]
        y = [node.y for node in nodes + int_nodes + int_gridX + more_nodes]
        ListTool.distinct(x)
        ListTool.distinct(y)
        return x, y
    
    def _intersection_nodesXY(self, edges: list[Edge], nodes: list[Node]) -> list[Node]:
        """用顶点坐标切割边"""
        intersections = []
        for node in nodes:
            nextL, nextR, nextU, nextD = (None,)*4  # 只切割上下左右最近的边
            minL, minR, minU, minD = (Const.MAX_VAL,)*4
            for edge in edges:
                if (
                    abs(edge.s.x - edge.e.x) < Const.TOL_DIST
                    or abs(edge.s.y - edge.e.y) < Const.TOL_DIST
                ):
                    continue  # 横平竖直不用切
                new_node= edge.intersection(Line(node,Vec3d(0,1,0)))
                if p is not None and p > 0 + Const.TOL_VAL and p + Const.TOL_VAL < 1:
                    if (
                        new_node.y > node.y + Const.TOL_DIST
                        and new_node.y - node.y < minU
                    ):
                        minU = new_node.y - node.y
                        nextU = new_node
                    if (
                        new_node.y + Const.TOL_DIST < node.y
                        and node.y - new_node.y < minD
                    ):
                        minD = node.y - new_node.y
                        nextD = new_node
                new_node, p = edge.point_at(y=node.y)
                if p is not None and p > 0 + Const.TOL_VAL and p + Const.TOL_VAL < 1:
                    if (
                        new_node.x > node.x + Const.TOL_DIST
                        and new_node.x - node.x < minR
                    ):
                        minR = new_node.x - node.x
                        nextR = new_node
                    if (
                        new_node.x + Const.TOL_DIST < node.x
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
        if precision > Const.TOL_VAL:
            ibounds = [math.ceil(bound / precision) for bound in bounds]
        else:
            ibounds = [0] * 4
        x = [i * precision for i in range(ibounds[0], ibounds[2])]
        y = [i * precision for i in range(ibounds[1], ibounds[3])]
        nodesX, nodesY = [], []
        for edge in edges:
            if (
                abs(edge.s.x - edge.e.x) < Const.TOL_DIST
                or abs(edge.s.y - edge.e.y) < Const.TOL_DIST
            ):
                continue  # 横平竖直不用切
            for xi in x:
                new_node, p = edge.point_at(x=xi)
                if p is not None and p > 0 + Const.TOL_VAL and p + Const.TOL_VAL < 1:
                    nodesX.append(new_node)
            for yi in y:
                new_node, p = edge.point_at(y=yi)
                if p is not None and p > 0 + Const.TOL_VAL and p + Const.TOL_VAL < 1:
                    nodesY.append(new_node)
        return nodesX, nodesY

    @Timer
    def _get_01matrix(self, poly: Polygon, x: list[float], y: list[float]) -> np.ndarray:
        """计算每个cell是否在多边形内。cellInside[0,:]=cellInside[:,0]=cellInside[m,:]=cellInside[:,n]=False"""
        poly = Polygon(*poly.offset(dist=-Const.TOL_DIST).to_array())
        m, n = len(x), len(y)
        ptmat = np.zeros((m, n), dtype=bool)
        for i in range(m):
            for j in range(n):
                ptmat[i, j] = poly.covers(Node(x[i], y[j]))
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

    @Timer
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
            mini, minj, maxi, maxj = Const.MAX_VAL, Const.MAX_VAL, 0, 0
            for l in range(len(covered_points[k])):
                if (
                    covered_points[k][l].x < x[0]
                    or covered_points[k][l].y < y[0]
                    or covered_points[k][l].x > x[-1]
                    or covered_points[k][l].y > y[-1]
                ):
                    idx_covered_points[k] = (0, 0, 0, 0)
                    break
                flagi, i = ListTool.bsearch(x, covered_points[k][l].x)
                flagj, j = ListTool.bsearch(y, covered_points[k][l].y)
                mini, minj, maxi, maxj = (
                    min(mini, i),
                    min(minj, j),
                    max(maxi, i),
                    max(maxj, j),
                )
            else:
                idx_covered_points[k] = (mini, minj, maxi, maxj)
        return idx_covered_points

    @Timer
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
        maxs = -Const.MAX_VAL
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
    def _postprocess(self) -> None: ...

class BreakEdgeAlgo(GeomAlgo):  # ✅
    """线段打断. 重叠部分会在端点处打断. 保持原线段的方向. 

    Args:
        edge_groups (list[Edge]|list[list[Edge]]): 待打断的一组线段. | 若干个分组，每组包含若干条线段.
    """
    def __init__(self,edge_groups:list[Edge]|list[list[Edge]]) -> None:
        super().__init__()
        if len(edge_groups)>0 and isinstance(edge_groups[0],Edge):
            self._is_group=False
            self.edge_groups=[edge_groups]        
        else:
            self.edge_groups=edge_groups
            self._is_group=True
        self.res=None
    def get_result(self)->list[Edge]|list[list[Edge]]:
        """获取打断的结果.

        Returns
        list[Edge]|list[list[Edge]]: 打断后的一组线段. | 若干个分组，每组包含打断后的线段，和打断前的分组一致.
        """
        if len(self.edge_groups)==0: return []
        self._preprocess()
        break_points=self._get_break_points()
        self.res=self._rebuild_lines(break_points)
        self._postprocess()
        return self.res
    def _preprocess(self)->None:
        """预处理"""
        super()._preprocess()
        # 去除0线段
        self.edge_groups=[[edge for edge in group if not edge.is_zero()] for group in self.edge_groups]
    def _postprocess(self)->None:
        """后处理"""
        if not self._is_group:
            self.res=self.res[0]
        super()._postprocess()
    def _get_break_points(self)->dict[Edge:list[Node]]:
        """获取线段上的断点，没有排序，也没有去重"""
        all_edges=sum(self.edge_groups,[])
        visited={line:set() for line in all_edges} # 记录已经求过交点的线段
        break_points={line:[line.s,line.e] for line in all_edges} # 记录线段上的断点
        rt=STRTree(all_edges)
        for line in all_edges:
            neighbors=rt.query(line.get_aabb())
            for other in neighbors:
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
    def _get_break_points_by_scanning(self)->dict[Edge:list[Node]]:  # TODO
        """获取线段上的断点，没有排序，也没有去重"""
        all_edges=sum(self.edge_groups,[])
        event_q=PriorityQueue()
        def cmp(a:Node,b:Node)->int:
            if a.y>b.y or a.y==b.y and a.x>b.x: return 1
            elif a.y<b.y or a.y==b.y and a.x<b.x: return -1
            else: return 0
        with ComparerInjector(Node,cmp,override_ops=True):
            for edge in all_edges:
                event_q.put(edge.s,edge.e)
            break_points=...
        return break_points    
    def _rebuild_lines(self,break_points:dict[Edge,list[Node]])->list[list[Edge]]:
        """根据断点重构线段"""
        broken_lines=[]
        for group in self.edge_groups:
            new_group=[]
            for line in group:
                break_points[line].sort(key=lambda p:line.get_param(p))
                pre=break_points[line][0]
                for p in break_points[line]:
                    if p.equals(pre): continue
                    new_group.append(line.slice_between(pre, p))
                    pre=p
            broken_lines.append(new_group)
        return broken_lines

class MergeEdgeAlgo(GeomAlgo):  # ✅
    """合并重叠的线段。使用Const.TOL_DIST判断区间端点精度。

    Args:
        edges (list[Edge]): 待合并的线段.
        preserve_intersection (bool, optional): 是否在交点处打断. Defaults to False.
        compare (Callable[[Edge,Edge],int], optional): 线段的优先级==0(等于)|==1(大于)|==-1(小于); 合并时保留较大的. Defaults to None (==0).
    """    
    def __init__(self,edges:list[Edge],break_at_intersections:bool=False,compare:Callable[[Edge,Edge],int]=None) -> None:
        super().__init__()
        self.edges=edges[:]
        self.merged:list[Edge]=[]
        self.break_at_intersections=break_at_intersections
        self.compare=compare
    def get_result(self)->list[Edge]:
        """获取合并后的线段.

        Returns:
            list[Edge]: 合并后的线段.
        """
        self._preprocess()
        # 去除0线段
        self.edges=list(filter(lambda edge: not edge.is_zero(),self.edges))
        # 分类
        lines:list[LineSeg]=[]
        arcs:list[Arc]=[]
        for edge in self.edges:
            if isinstance(edge,LineSeg): lines.append(edge)
            if isinstance(edge,Arc): arcs.append(edge)
        # 按平行共线分组
        parallel_line_groups=self._group_parallel_lines(lines)
        parallel_arc_groups=self._group_parallel_arcs(arcs)
        collinear_line_groups=[]
        for group in parallel_line_groups:
            collinear_line_groups+=self._group_collinear_from_parallel_lines(group)
        collinear_arc_groups=[]
        for group in parallel_arc_groups:
            collinear_arc_groups+=self._group_collinear_from_parallel_arcs(group)            
        # 3.合并重叠的线段
        with ComparerInjector(Edge,self.compare,override_ops=True):
            for group in collinear_line_groups:
                self.merged+=self._merge_collinear_lines(group)
            for group in collinear_arc_groups:
                self.merged+=self._merge_collinear_arcs(group)
        # 4.后处理
        self._postprocess()
        return self.merged
    def _preprocess(self)->None:
        """前处理"""
        super()._preprocess()
    def _postprocess(self)->None:
        """后处理"""
        # 按需打断
        if self.break_at_intersections:
            self.merged=BreakEdgeAlgo([self.merged]).get_result()[0]
        super()._postprocess()        
    def _group_parallel_lines(self,lines:list[LineSeg])->list[list[LineSeg]]:  # ✅OK
        """线段按角度分组"""
        # 先将线段角度转换到[0,pi)范围内
        for i,line in enumerate(lines):
            if Const.cmp_ang(line.angle,math.pi)>=0 and Const.cmp_ang(line.angle,math.pi*2)<0:
                lines[i]=line.opposite()
        # 角度在误差范围内的分为一组
        line_groups=[]
        lines.sort(key=lambda line:line.angle)
        current_angle=-Const.MAX_VAL
        for line in lines:
            if line.angle-current_angle>Const.TOL_ANG:  # !parallel
                new_group=[line]
                line_groups.append(new_group)
                current_angle=line.angle
            else: 
                new_group.append(line)
        return line_groups
    def _group_parallel_arcs(self,arcs:list[Arc])->list[list[Arc]]:  # ✅OK
        """圆弧按圆心分组"""
        # 先将圆弧转换为逆时针
        for i,arc in enumerate(arcs):
            if arc.bulge<0:
                arcs[i]=arc.opposite()
        # 圆心在误差范围内的分为一组
        centers:list[Node]=[]
        center_dict:dict[Node,list[Arc]]={}
        for arc in arcs:
            center=GeomUtil.find_or_insert_node(arc.center,centers)
            if center not in center_dict: 
                center_dict[center]=[arc]
            else:
                center_dict[center].append(arc)
        arc_groups=list(center_dict.values())
        return arc_groups
    def _group_collinear_from_parallel_lines(self,lines:list[LineSeg]):
        """将平行直线按共线分组"""
        # 找一根最长的，作为方向向量
        longest_edge=max(lines,key=lambda edge:edge.length)
        # 沿法向量（右转90度）排序
        unit_vector=longest_edge.to_vec3d().unit() # 单位向量
        normal_vector=unit_vector.cross(Vec3d(0,0,1)) # 法向量
        lines.sort(key=lambda line:line.s.to_vec3d().dot(normal_vector))
        # 分组
        collinear_groups:list[list[LineSeg]]=[]
        current_dist=-Const.MAX_VAL
        for line in lines:
            dist_s=line.s.to_vec3d().dot(normal_vector) # 投影
            if dist_s-current_dist>Const.TOL_DIST: # !collinear
                new_group=[line]
                collinear_groups.append(new_group)
                current_dist=dist_s
            else: 
                dist_e=line.e.to_vec3d().dot(normal_vector)
                if abs(dist_e-current_dist)<Const.TOL_DIST:  # 端点投影距离也在范围内的才算共线
                    new_group.append(line)
                else: 
                    collinear_groups.append([line])
        return collinear_groups
    def _group_collinear_from_parallel_arcs(self,arcs:list[Arc]):
        """将平行圆弧按共圆分组"""
        # 按半径排序
        arcs.sort(key=lambda arc:arc.radius)
        # 分组
        collinear_groups:list[list[Arc]]=[]
        current_radius=0
        for arc in arcs:
            if arc.radius-current_radius>Const.TOL_DIST: # !collinear
                new_group=[arc]
                collinear_groups.append(new_group)
                current_radius=arc.radius
            else:
                new_group.append(arc)
        return collinear_groups
    def _merge_collinear_lines(self,lines:list[LineSeg]):
        """合并共线的线段"""
        # 找一根最长的作为基准，全部按照arc_length param构建为Interval1d，然后合并
        longest=max(lines,key=lambda line:line.length)
        proj=lambda p:longest.get_param(p,arc_length=True)
        intvs=[Interval1d(proj(line.s),proj(line.e),line) for line in lines]
        if self.compare is None:
            merged_intvs=Interval1d.union(intvs)
        else:
            merged_intvs=Interval1d.union(intvs,ignore_value=False)
            # merged_intvs=SegmentTree(intervals).get_united_leaves()  # 用线段树
        merged_lines=[]
        d=longest.length
        for intv in merged_intvs:
            merged_lines.append(longest.slice_between(longest.point_at(intv.l/d),longest.point_at(intv.r/d)))
        return merged_lines
    def _merge_collinear_arcs(self,edges:list[Arc]):
        """合并共线的圆弧"""
        # 找一根最长的作为基准，全部构建为Interval1d，然后合并；
        # 构建的时候注意可能出现跨越0点合并的情况，因此除了本身就
        # 跨越0点的区间不需要之外，其他都需要在(-2pi,0]区间和[0,2pi)区间内构各建1次；
        # 合并完之后，从0开始扫描角度为的2pi区间；如果有一段>2pi的弧，就构造一个圆；
        baseline=max(edges,key=lambda edge:edge.length)
        cir=baseline.radius*2*math.pi
        intvs=[]
        for edge in edges:
            l=baseline.get_param(edge.s,arc_length=True)
            r=baseline.get_param(edge.e,arc_length=True)
            if r<l:  # 跨越0点，只用构建1次
                intvs.append(Interval1d(l-cir,r,edge))
            else:  # 没跨越0点，要构建2次
                intvs.append(Interval1d(l,r,edge))
                intvs.append(Interval1d(l-cir,r-cir,edge))
        if self.compare is None:
            merged_intvs=Interval1d.union(intvs)
        else:
            merged_intvs=Interval1d.union(intvs,ignore_value=False)
            # merged_intvs=SegmentTree(intervals).get_united_leaves()  # 用线段树
        # 整圆的情况
        if len(merged_intvs)==1 and Const.cmp_dist(merged_intvs[0].r-merged_intvs[0].l,cir)>=0:
            return [Circle(baseline.center,baseline.radius)]
        merged_edges=[]
        d=baseline.length
        for start_i,intv in enumerate(merged_intvs):
            if Const.cmp_dist(intv.l,0)<=0 and Const.cmp_dist(intv.r,0)>0:
                start_param=intv.l
                break
        for intv in merged_intvs[start_i:]:
            if intv.r-start_param>cir: break
            merged_edges.append(baseline.slice_between(baseline.point_at(intv.l/d),baseline.point_at(intv.r/d)))
        return merged_edges

class FindConnectedGraphAlgo(GeomAlgo):
    """求连通图.

    Args:
        edges (list[Edge]): 所有线段，需要打断.
    """
    def __init__(self,lines:list[Edge]) -> None:
        super().__init__()
        self.lines=lines
        self.connected_graphs:list[list[Edge]]=[]
    def get_result(self)->list[list[Edge]]:
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
                neighbor_lines=rt.query(node.get_aabb())
                for line in neighbor_lines:
                    if line not in visited_lines:
                        if node.dist(line.s)<Const.TOL_DIST or node.dist(line.e)<Const.TOL_DIST:
                            q.append(line)
                            visited_lines.add(line)
        return res
    def _preprocess(self)->None:
        # 去除0线段
        self.lines=list(filter(lambda line: not line.is_zero(),self.lines))
    def _postprocess(self)->None: ...

class FindOutlineAlgo(GeomAlgo):
    """求单个连通图形的外轮廓

    Args:
        edges (list[Edge]): 所有线段，无需打断.
    
    Todo:
        * 圆弧.
    """    
    def __init__(self,edges:list[Edge]) -> None:
        super().__init__()
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
        """找起始边：先找x最小的点，然后向右出发"""
        start_edge=min(self.edges,key=lambda edge:edge.get_aabb().minx)
        box=start_edge.get_aabb()
        left_bound=LineSeg(box.lb(),box.lt())
        intersections=start_edge.intersection(left_bound)
        if len(intersections)==0:
            start_node=start_edge.s
        else:
            start_node=intersections[0]
        return LineSeg(Node(start_node.x-Const.TOL_DIST*2,start_node.y),start_node)
    def _find_outline(self,rt_edges:STRTree ,start_edge:Edge)->Loop: 
        """顺着start_edge逆时针找一圈外轮廓"""
        outline:list[Edge]=[]
        pre_edge=start_edge
        this_node=pre_edge.e
        while True: # 每次循环从pre_edge出发，找下一条边，直到回到起点
            # 搜索与当前出发点临近的边
            nearest_edges=rt_edges.query(this_node.get_aabb()) 
            # 求当前顶点到这些边的端点的连线的集合
            for edge in nearest_edges:
                if not this_node.is_on_edge(edge): continue
                if not this_node.equals(edge.s):
                    GeomUtil.add_edge_to_node_in_order(this_node,edge.opposite().slice_between(this_node,edge.s))
                if not this_node.equals(edge.e):
                    GeomUtil.add_edge_to_node_in_order(this_node,edge.slice_between(this_node,edge.e))
            # 从this_node出发，找到pre_edge的下一条边
            new_edge=GeomUtil.find_next_edge_out(this_node,pre_edge)
            # 求所有与new_edge可能相交的线
            nearest_edges=rt_edges.query(new_edge.get_aabb())
            # 遍历相交的线，取距离最近的一个交点(起点除外)，作为下一个顶点
            min_param_dist=1
            next_node=new_edge.e
            for nearest_edge in nearest_edges:
                intersections=new_edge.intersection(nearest_edge)
                for p in intersections:
                    d=new_edge.get_param(p)
                    if not p.equals(this_node) and d<min_param_dist and p.is_on_edge(new_edge): # 不是起点，且距离最近
                        min_param_dist=d
                        next_node=p
            new_edge=new_edge.slice_between(this_node,next_node)
            if len(outline)>0 and new_edge.equals(outline[0]): # 回到起点就结束
                break
            else: # 保存当前边，继续找下一条边
                outline.append(new_edge)
                pre_edge=new_edge
                this_node=next_node
        nodes=[edge.s for edge in outline]
        bulges=[edge.bulge if isinstance(edge,Arc) else 0 for edge in outline ]
        return Loop(nodes,bulges)
    def _postprocess(self) -> None:
        pass

class FindLoopAlgo(GeomAlgo):  # ✅
    """重建曲线所围成的区域的几何拓扑.

    Args:
        edges (list[Edge]): 所有曲线，无需打断.
        directed (bool, optional): 输入的边集是否有向. Defaults to False (无向图).
        cancel_out_opposite (bool, optional): 是否去除重合的反向边. Defaults to False. 参数对于无向图(!directed)不生效.
    """
    def __init__(
        self,
        edges: list[Edge],
        directed: bool = False,
        cancel_out_opposite: bool = False,
    ) -> None:
        super().__init__()
        self.edges = edges
        self.directed = directed
        self.cancel_out_opposite = cancel_out_opposite and directed
        self.loops = []
    # @Timer
    def _preprocess(self)->None:
        super()._preprocess()
        # 打断边
        self.edges=BreakEdgeAlgo([self.edges]).get_result()[0]
        # 构建图结构
        self.edges=list(filter(lambda edge:not edge.is_zero(),self.edges))
        self.nodes:list[Node]=[]
        for edge in self.edges:
            s=GeomUtil.find_or_insert_node(edge.s,self.nodes)
            e=GeomUtil.find_or_insert_node(edge.e,self.nodes)
            edge.s,edge.e=s,e
            # 处理重合的反向边：有反向边就pop出去；没有就把edge添加进来
            if (not self.cancel_out_opposite
                or self.cancel_out_opposite and self._pop_opposite(edge) is None
            ):
                GeomUtil.add_edge_to_node_in_order(s,edge)
            if not self.directed:  # 无向图
                GeomUtil.add_edge_to_node_in_order(e,edge.opposite())
        if self.directed:  # 过滤掉被pop出去的边
            self.edges=sum([node.edge_out for node in self.nodes],[])
    def _pop_opposite(self,edge:Edge):
        op=edge.opposite()
        i=ListTool.first(edge.e.edge_out,cond=lambda x:x==op)
        return edge.e.edge_out.pop(i) if i!=-1 else None
    def get_result(self)->list[Loop]:
        self._preprocess()
        self.loops=self._find_loop()
        self._postprocess()
        return self.loops
    # @Timer
    def _postprocess(self)->None:
        super()._postprocess()
    # @Timer
    def _find_loop(self)->list[Loop]:
        # 沿逆时针优先的方向，顺着边往下找，
        # 碰到已找过的边就计为一个环，直到所有边都找完。
        loops:list[Loop] =[]
        edge_stack:list[Edge]=[]  # 当前栈里的边
        edge_num=len(self.edges) if self.directed else 2*len(self.edges)
        while edge_num>0:  # 每次循环找一个环，直到所有边都被遍历过
            if len(edge_stack)==0:  # 栈空了就随便找一条边作为起始
                for node in self.nodes:
                    if len(node.edge_out)>0:
                        edge_stack=[node.edge_out[0]]
                        break
            while True:  # 以pre.e为当前结点开始找一个环
                node=edge_stack[-1].e  # 当前结点
                next_edge=GeomUtil.find_next_edge_out(node, edge_stack[-1])  # 按角度和半径找下一条出边，确保内环优先逆时针方向
                if next_edge in edge_stack:  # 如果找到了已访问的边就封闭这个环
                    start_idx=edge_stack.index(next_edge)  # 环的起始边
                    new_loop=edge_stack[start_idx:]
                    loops.append(Loop.from_edges(new_loop))  # 先将此环加入结果list
                    edge_stack=edge_stack[:start_idx]  # 剩余的留在栈里继续下一轮
                    for edge in new_loop:  # 并把环上的边从邻接表和栈里都删掉
                        edge.s.edge_out.remove(edge)
                    edge_num-=len(new_loop)  # 从总边数中减去环的边数
                    break
                else:  # 如果找到的不是已访问的边，就将此边加入栈，接着找下一条边
                    edge_stack.append(next_edge)
        return loops

class BooleanOperation:
    """多边形布尔运算"""
    @classmethod
    def union(cls,geoms:list[Polygon])->list[Polygon]:
        """布尔并.

        Args:
            geoms (list[Polygon]): 待合并的多边形.

        Returns:
            list[Polygon]: 并集.
        """
        # 正负配对，只取最外层的
        all_loops=sum([list(geom.all_loops) for geom in geoms],[])
        cond=lambda depth:depth==1
        return cls._rebuild_loop_topology(all_loops,condition=cond)
    @classmethod
    def intersection(cls,geoms:list[Polygon])->list[Polygon]:
        """布尔交.

        Args:
            geoms (list[Polygon]): 要求交的多边形.

        Returns:
            list[Polygon]: 交集.
        """
        # 正负配对，只取第N层的
        all_loops=sum([list(geom.all_loops) for geom in geoms],[])
        cond=lambda depth:depth==len(geoms)
        return cls._rebuild_loop_topology(all_loops,condition=cond)
    @classmethod
    def difference(cls,a:Polygon|list[Polygon],b:Polygon|list[Polygon])->list[Polygon]:
        """布尔差 (a-b). 入参为list时会先合并.

        Args:
            a (Polygon|list[Polygon]): 被减数.
            b (Polygon|list[Polygon]): 减数.

        Returns:
            list[Polygon]: 差集.
        """
        if isinstance(a,list): a=cls.union(a)
        if isinstance(b,list): b=cls.union(b)
        # 反转b的方向，然后求并
        all_loops=[loop for plg in a for loop in plg.all_loops]
        all_loops.extend([loop.reversed() for plg in b for loop in plg.all_loops])
        cond=lambda depth:depth==1
        return cls._rebuild_loop_topology(all_loops,condition=cond)
    @classmethod
    def _loop2polygon(cls,rebuilt_loops:list[Loop],condition:Callable[[int],bool]=None)->list[Polygon]:
        """环按照覆盖关系组成多边形"""
        condition=condition or (lambda _:True)
        root=cls._build_loop_tree(rebuilt_loops)
        polygons=[]
        cls._traverse_loop_tree(
            root=root,
            depth=0,
            condition=condition,
            stack=[],
            out_polygons=polygons,
        )
        return polygons
    @classmethod
    def _rebuild_loop_topology(cls,loops:list[Loop],condition:Callable[[int],bool]=None)->list[Polygon]:
        """重建所有环的拓扑，并按条件组合成Polygon.

        Args:
            loops (list[Loop]): 待重建的环.
            condition (Callable[[int],bool]): 组合成多边形的条件. Defaults to (depth:int)->True.

        Returns:
            list[Polygon]: 配对的环组成的多边形.
        """
        condition=condition or (lambda _:True)
        all_edges=sum([list(loop.edges) for loop in loops],[])
        rebuilt_loops=FindLoopAlgo(all_edges,directed=True,cancel_out_opposite=True).get_result()
        return cls._loop2polygon(rebuilt_loops,condition)
    @classmethod
    def _sort_cover_tree(cls,i:TreeNode[Loop]):
        """对于父子完全重合的loop，根据爷爷的方向，调整正负关系"""
        # 完全重叠时，要尽可能使相邻层级不同向：
        # 当爸爸和爷爷同向时，就检查所有的儿子，如果有重叠且方向不同的，就交换爸爸和儿子
        # ++- -> +-+ 或 --+ -> -+-
        ai=i.obj and i.obj.area
        ap=i.parent and i.parent.obj and i.parent.obj.area
        if ai is not None and (ap is None or ap>0)^(ai<0):
            for j in i.child:
                aj=j.obj.area
                if abs(ai)-abs(aj)<Const.TOL_AREA and j.obj.covers(i.obj) and (ai>0)^(aj<0):
                    i.obj,j.obj=j.obj,i.obj
                    break
        for j in i.child: cls._sort_cover_tree(j)
    @classmethod
    def _build_loop_tree(cls,loops:list[Loop])->TreeNode[Loop]:
        """根据覆盖关系构建树.

        Args:
            loops (list[Loop]): 要求不得self-cross，也不能互相cross；否则需要先执行FindLoopAlgo.

        Returns:
            TreeNode[Loop]: 虚拟的树根，root.obj=None.
        """
        # 按面积排序，从小到大找到的第一个能覆盖自己的，就是自己的parent
        loops.sort(key=lambda loop:abs(loop.area))
        t =[TreeNode(loop) for loop in loops]
        for i in range(1,len(t)):
            for j in range(i):
                if t[j].parent is not None: continue
                if t[i].obj.covers(t[j].obj): t[j].parent=t[i]
        root=TreeNode(None)
        for i in t:
            if i.parent is None: i.parent=root
            i.parent.child.append(i)
        # 对于父子完全重合的loop，根据爷爷的方向，调整正负关系
        cls._sort_cover_tree(root)
        return root    
    @classmethod
    def _traverse_loop_tree(cls,root:TreeNode[Loop],depth:int,condition:Callable[[int],bool],stack:list[TreeNode],out_polygons:list[Polygon])->list[Loop]:
        """遍历Loop的覆盖关系树，按条件返回配对关系.

        Args:
            root (TreeNode[Loop]): 当前结点.
            depth (int): 当前结点的深度.
            condition (Callable[[int],bool]): 配对条件:(depth:int)->bool.
            stack (list[TreeNode]): 当前loop栈.
            out_polygons (list[Polygon]): 配对的Polygon.

        Returns:
            list[TreeNode]: 配对的后代.
        """
        # 按照括号配对原则：
        # 1. 遇shell则depth+1入栈
        # 2. 遇hole则寻找最近的且depth相同的祖先shell配对，然后depth-1不入栈
        # 返回满足condition的shells，组成polygon
        root.depth=depth
        root.holes=[]
        for child in root.child:
            if child.obj.area>0:  # shell
                stack.append(child)
                cls._traverse_loop_tree(child,depth+1,condition,stack,out_polygons)
                stack.pop()
            else:  # hole
                for i in range(len(stack)-1,-1,-1):
                    if stack[i].depth==depth:
                        stack[i].holes.append(child.obj)
                        break
                cls._traverse_loop_tree(child,depth-1,condition,stack,out_polygons)
        if condition(depth) and root.obj is not None and root.obj.area>0:
            out_polygons.append(Polygon(shell=root.obj, holes=root.holes, make_valid=False))
