import math
from dataclasses import dataclass
from lib.geom import Geom,Node,LineSeg,Loop,Polygon
from lib.geom_algo import BooleanOperation,MergeEdgeAlgo,FindLoopAlgo
from lib.biz_algo import BizAlgo
from lib.utils import Constant
from lib.geom_plotter import CADPlotter

import shapely

INF=math.inf

@dataclass
class DrainingRegion:
    loop: Loop
    nearest_drain: Node=None
    dist:float=math.inf
class FindDrainingPlan(BizAlgo):
    def __init__(self,roofs:list[Polygon],drain_points:list[Node],cut_tol:float=300)->None:
        super().__init__()
        self.roofs:list[Polygon]=roofs
        self.drain_points:list[Node]=drain_points
        self.roof_drain:dict[Polygon:list[Node]]={}
        self.cut_tol=cut_tol
        """屋面区域->区域包含的落水口"""
        self.good_regions:list[Polygon]=[]
        """布置好排水的区域"""
        self.bad_regions:list[Polygon]=[]
        """没布置好的区域"""
    def _postprocess(self) -> None:
        ...
    def _preprocess(self) -> None:
        # 屋面-落水口点位配对
        self.roof_drain:dict[Polygon,list[Node]]={roof:[] for roof in self.roofs}
        for drain_point in self.drain_points:
            for roof in self.roofs:
                if roof.covers(drain_point):
                    self.roof_drain[roof].append(drain_point)
                    break
            else:
                min_dist=float("inf")
                for roof in self.roofs:
                    pt=roof.closest_point(drain_point)
                    if dist:=drain_point.dist(pt)<min_dist:
                        min_dist,closest_pt=dist,pt
                self.roof_drain[roof].append(closest_pt)
    def get_result(self):
        self._preprocess()
        for roof,drains in self.roof_drain.items():
            # 划分子区域

            roof_edges=sum([list(loop.edges) for loop in roof.all_loops],[])
            cut_edges=self._get_cut_edges(roof,drains)
            
            cut_loops=FindLoopAlgo(roof_edges+cut_edges).get_result()
            cut_loops=list(filter(lambda loop:loop.area>0 and roof.covers(loop),cut_loops))

            center_points=[loop.get_centroid() for loop in cut_loops]
            regions={c:DrainingRegion(loop) for c,loop in zip(center_points,cut_loops)}

            # 子区域中心点+落水口，构建可见性图
            vis_graph=self._build_vis_graph(center_points+drains,roof)

            # 计算最近的落水口
            for drain in drains:
                dist_to,_=self._get_shortest_path(vis_graph,drain)
                for node in center_points:
                    if dist_to[node]<regions[node].dist:
                        regions[node].dist=dist_to[node]
                        regions[node].nearest_drain=drain

            # 计算各落水口汇水区域，和总加权汇水长度
            base_volume=sum([reg.dist*reg.loop.area for reg in regions.values()])
            base_unions:dict[Node,Polygon]={}
            for drain in drains:
                loops_of_drain=[reg.loop for reg in regions.values() if reg.nearest_drain is drain]
                u=BooleanOperation._pair_loops(loops_of_drain[drain])
                assert len(u)==1
                base_unions[drain]=u[0]

            # 判断可以一刀切的边
            safe_edges=[]
            new_safe_edges=[]
            for edge in cut_edges:
                for poly in base_unions.values():
                    if Geom.GeomRelation.Inside in poly.shell._relation_with_edge(edge):
                        break
                else: new_safe_edges.append(edge)
            vis_graph=self._update_vis_graph(vis_graph,new_safe_edges)
            safe_edges+=new_safe_edges

            # 落水口、cut_edges分配到切完的屋面
            # 1个屋面只有1个落水口：safe
            # 对于有多个落水口的屋面，枚举cut_edge，计算切完之后的加权汇水长度，取最小的方案
            # 继续往下切，直到每个屋面一个落水口为止
            # 计算每个汇水区域的汇水路径和汇水线
            # 计算汇水方向
            # 对于坡过长的汇水区域，寻找过水洞位置
            # 切屋面、计算汇水线

        self._postprocess()
        return self.good_regions,self.bad_regions
    
    def _triangulate_drains(self,drains:list[Node])->list[LineSeg]:
        if len(drains)<=1: return []
        if len(drains)==2: return [LineSeg(*drains)]
        points=shapely.MultiPoint([[node.x,node.y] for node in drains])
        lines=shapely.delaunay_triangles(points,tolerance=self.const.TOL_DIST,only_edges=True)
        edges=[LineSeg(Node(*line.coords[0]),Node(*line.coords[1])) for line in lines.geoms]
        merged=MergeEdgeAlgo(edges,break_at_intersections=True).get_result()
        return merged

    def _get_cut_edges(self,roof:Polygon,drains:list[Node])->list[Loop]:
        """将屋面横竖划分为若干子区域；划分位置：边界顶点、落水口平分线"""
        bounds=roof.get_mbb()
        cut_edges=[]
        # 用区域顶点切分
        cutX,cutY=[],[]
        for loop in roof.all_loops:
            for i,node in enumerate(loop.nodes):
                if loop.edge(i-1).tangent_at(0).angle_between(loop.edge(i).tangent_at(0))<self.const.TOL_ANG: continue
                x_line=LineSeg(Node(node.x,bounds[0].y),Node(node.x,bounds[1].y))
                x_segs=roof.clips_edge(x_line)
                for seg in x_segs:
                    if seg.s==node or seg.e==node:
                        cut_edges.append(seg)
                        cutX.append(node.x)
                y_line=LineSeg(Node(bounds[0].x,node.y),Node(bounds[1].x,node.y))
                y_segs=loop.clips_edge(y_line)
                for seg in y_segs:
                    if seg.s==node or seg.e==node:
                        cut_edges.append(seg)
                        cutY.append(node.y)
        # 用落水口平分线切分
        cutX_mid,cutY_mid=[],[]
        drain_triangles=self._triangulate_drains(drains)
        for edge in drain_triangles:
            if math.pi/4<edge.angle_of_line<math.pi/4*3:  # 连线为竖直方向，做水平切分
                new_y=edge.point_at(0.5).y
                for y in cutY: 
                    if abs(y-new_y)>self.cut_tol: 
                        cutY_mid.append(new_y)                
            else:  # 连线为水平方向，做竖直切分
                new_x=edge.point_at(0.5).x
                for x in cutX: 
                    if abs(x-new_x)>self.cut_tol: 
                        cutX_mid.append(new_x)
        cut_edges+=[LineSeg(Node(x,bounds[0].y),Node(x,bounds[1].y)) for x in cutX_mid]
        cut_edges+=[LineSeg(Node(bounds[0].x,y),Node(bounds[1].x,y)) for y in cutY_mid]
        return cut_edges

    def _get_shortest_path(vis_graph:dict[Node,list[Node]],source:Node) -> tuple[dict[Node,float],dict[Node,Node]]:
        """dijkstra.
            d: 各节点最短路径长度
            pre: 各节点最短路径上的前驱节点
        """
        d={node:math.inf for node in vis_graph}
        d[source]=0
        pre={}
        safe_set=set()
        for i in range(len(vis_graph)-1):
            min_d=math.inf
            for node in vis_graph:
                if node not in safe_set and d[node]<min_d:
                    min_d,safe_node=d[node],node
            if math.isinf(min_d): break
            safe_set.add(safe_node)
            for node in vis_graph[safe_node]:
                if d[node]>d[safe_node]+safe_node.dist(node):
                    d[node]=d[safe_node]+safe_node.dist(node)
                    pre[node]=safe_node
        return d,pre

if __name__=="__main__":
    import json
    from tool.converter.json_converter import JsonLoader
    with open("apps/find_draining_plan/case_1.json",'r') as f:
        geoms=json.load(f,object_hook=JsonLoader.from_cad_obj)
    roof_boundaries=[geom for geom in geoms if isinstance(geom,Loop)]
    drain_points=[geom for geom in geoms if isinstance(geom,Node)]
    roofs=BooleanOperation._pair_loops(roof_boundaries)
    res=FindDrainingPlan(roofs,drain_points).get_result()



    # 45度切割法
    # def get_result(self):
    #     self._preprocess()
    #     for roof in self.roofs:
    #         self.cut_region(roof,self.roof_drain[roof])
    #     for region in self.bad_regions:
    #         new_hole=self.find_hole(region)
    #         if new_hole is not None:
    #             self.roof_drain[region]=self.roof_drain.get(roof,[])+[new_hole]
    #             self.cut_region(region,new_hole)
    #     for region in self.good_regions:
    #         self.cut_dead_corner(region)
    #     self._postprocess()
    #     return self.good_regions,self.bad_regions
    # def cut_region(self,region:Polygon,drains:list[Node])->None:
    #     ...
    # def find_hole(self,region:Polygon)->Node:
    #     ...
    # def cut_dead_corner(self,region:Polygon)->None:
    #     ...    

