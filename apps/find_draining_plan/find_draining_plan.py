import math
from dataclasses import dataclass
from lib.geom import Node,LineSeg,Loop,Polygon,GeomRelation,Polyedge
from lib.geom_algo import BooleanOperation, FindLoopAlgo, MergeEdgeAlgo
from lib.biz_algo import BizAlgo
from lib.geom_plotter import CADPlotter

import shapely

INF=math.inf

@dataclass
class DrainingRegion:
    reg: Polygon
    target_drain: Node=None
    dist:float=math.inf
    sub_regions: list["DrainingRegion"]=None

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
        ...
    def _match_roof_drain(self,roofs:Polygon,drain_points:list[Node])->dict[Polygon,list[Node]]:
        """屋面区域-落水口点位配对"""
        roof_drain:dict[Polygon,list[Node]]={roof:[] for roof in roofs}
        for drain_point in drain_points:
            for roof in roofs:
                if roof.covers(drain_point):
                    roof_drain[roof].append(drain_point)
                    break
            else:
                min_dist=float("inf")
                for roof in roofs:
                    pt=roof.closest_point(drain_point)
                    if dist:=drain_point.dist(pt)<min_dist:
                        min_dist,closest_pt=dist,pt
                roof_drain[roof].append(closest_pt)
        return roof_drain
    def _cut_roof(self,roof:Polygon,drains:list[Node])->list[DrainingRegion]:
        """将屋面切分为若干汇水区域，和落水口一一对应"""
        
        # 划分网格
        roof_edges=sum([list(loop.edges) for loop in roof.all_loops],[])
        cut_edges=self._get_cut_edges(roof,drains)
        dual_edges=[edge.opposite() for edge in cut_edges]

        cut_loops=FindLoopAlgo(roof_edges+cut_edges+dual_edges,directed=True).get_result()
        # cut_loops=FindLoopAlgo(roof_edges+cut_edges,directed=False).get_result()
        # cut_loops=list(filter(lambda loop:loop.area>0 and roof.covers(Polygon(loop,make_valid=False)),cut_loops))

        center_points=[loop.get_centroid() for loop in cut_loops]
        regions={c:DrainingRegion(Polygon(loop)) for c,loop in zip(center_points,cut_loops)}

        # 构建可见性图
        vis_graph=self._build_vis_graph(roof,center_points,drains)

        # 计算最近的落水口
        for drain in drains:
            dist_to,pre=self._get_shortest_path(vis_graph,drain)
            for node in center_points:
                if dist_to[node]<regions[node].dist:
                    regions[node].dist=dist_to[node]
                    # path=[node]
                    # while path[-1] is not drain:
                    #     path.append(pre[path[-1]])
                    # regions[node].path=Polyedge(path)
                    regions[node].target_drain=drain

        # 计算各落水口汇水区域，和总加权汇水长度
        base_volume=sum([reg.dist*reg.reg.area for reg in regions.values()])
        base_unions:dict[Node,Polygon]={}
        for drain in drains:
            loops_of_drain=[reg.reg for reg in regions.values() if reg.target_drain is drain]
            u=BooleanOperation._pair_loops(loops_of_drain)
            assert len(u)==1
            base_unions[drain]=u[0]

        # 判断可以一刀切的边
        safe_edges=[]
        new_safe_edges=[]
        for edge in cut_edges:
            for poly in base_unions.values():
                if GeomRelation.Inside in poly.shell._relation_with_edge(edge):
                    break
            else: new_safe_edges.append(edge)
        vis_graph=self._update_vis_graph(vis_graph,new_safe_edges)
        safe_edges+=new_safe_edges

        # 落水口、cut_edges分配到切完的屋面
        dual_edges=[edge.opposite() for edge in safe_edges]
        new_roofs=FindLoopAlgo(roof_edges+safe_edges+dual_edges,directed=True).get_result()            
        new_roof_drain=self._match_roof_drain(new_roofs,drains)
        
        # 1个屋面只有1个落水口：safe
        remained_roof_drain:dict[Polygon, list[Node]]={}
        for roof,drains in new_roof_drain.items():
            if len(drains)==1:
                safe_roofs[drains[0]]=roof
            else:
                remained_roof_drain[roof]=drains

        # 对于有多个落水口的屋面，枚举cut_edge，计算切完之后的加权汇水长度，取最小的方案
        for edge in cut_edges:
            for 

        # 继续往下切，直到每个屋面一个落水口为止
        # 计算每个汇水区域的汇水路径和汇水线
        # 计算汇水方向
        # 对于坡过长的汇水区域，寻找过水洞位置
        # 切屋面、计算汇水线
        ...

    def get_result(self):
        self._preprocess()
        roof_drain=self._match_roof_drain(self.roofs,self.drain_points)
        safe_roofs=[]
        bad_roofs=[]
        for roof,drains in roof_drain.items():
            match len(drains):
                case 0: bad_roofs.append(roof)
                case 1: safe_roofs.append(DrainingRegion(roof,drains[0]))
                case _: safe_roofs+=self._cut_roof(roof,drains)
        for reg in safe_roofs: 
            self._get_converge_edges(reg,bad_roofs)
        for reg in bad_roofs:
            self._find_pass_for_bad_
        self._postprocess()
        return self.good_regions,self.bad_regions
    def _update_vis_graph(self,vis_graph:dict[Node,set[Node]],new_safe_edges:list[LineSeg]):
        for i,vis in vis_graph.items():
            for j in vis:
                for edge in new_safe_edges:
                    inter=LineSeg(i,j).intersection(edge)
                    if len(inter)>0 and inter[0]!=edge.s and inter[0]!=edge.e:  # 被切到的变为不可见
                        vis.remove(j)
                        vis_graph[j].remove(i)
                        break

    def _build_vis_graph(self,roof:Polygon,reg_centers:list[Node],drains:list[Node])->dict[Node,set[Node]]:
        """计算房间内可见性图"""
        roof_nodes=sum([loop.nodes for loop in roof.all_loops],[])
        vis_graph={node:set() for node in roof_nodes+reg_centers+drains}  # 可见性图（邻接表）
        # 加入可直达的边
        for i in roof_nodes+reg_centers:
            for j in roof_nodes+drains:
                vis_line=LineSeg(i,j)
                if roof.covers(vis_line):
                    vis_graph[i].add(j)
                    vis_graph[j].add(i)
        return vis_graph    
    
    def _triangulate_drains(self,drains:list[Node])->list[LineSeg]:
        if len(drains)<=1: return []
        if len(drains)==2: return [LineSeg(*drains)]
        points=shapely.MultiPoint([[node.x,node.y] for node in drains])
        lines=shapely.delaunay_triangles(points,tolerance=self.const.TOL_DIST,only_edges=True)
        edges=[LineSeg(Node(*line.coords[0]),Node(*line.coords[1])) for line in lines.geoms]
        merged=MergeEdgeAlgo(edges,break_at_intersections=True).get_result()
        return merged

    def _get_cut_edges(self,roof:Polygon,drains:list[Node])->list[LineSeg]:
        """将屋面横竖划分为若干子区域；划分位置：边界顶点、落水口平分线"""
        bounds=roof.get_mbb()
        cut_edges=[]
        # 1. 用区域顶点切分
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
                y_segs=roof.clips_edge(y_line)
                for seg in y_segs:
                    if seg.s==node or seg.e==node:
                        cut_edges.append(seg)
                        cutY.append(node.y)
        # 2. 用落水口平分线切分
        def add_cut_mid(new_t:float,cut:list[float],cut_mid:list[float]):
            for t in cut_mid: 
                if abs(t-new_t)<self.cut_tol:  # 有重复的平分线就不再加
                    break
            else:  # 没有重复的就加
                for t in cut:
                    if abs(t-new_t)<self.cut_tol:  # 附近有顶点就吸到顶点上
                        cut_mid.append(t)
                        break
                else: cut_mid.append(new_t)  # 附近没有顶点就直接加
        cutX_mid,cutY_mid=[],[]
        drain_triangles=self._triangulate_drains(drains)
        for edge in drain_triangles:
            if math.pi/4<edge.angle_of_line<math.pi/4*3:  # 连线为竖直方向，做水平切分
                new_y=edge.point_at(0.5).y
                add_cut_mid(new_y,cutY,cutY_mid)
            else:  # 连线为水平方向，做竖直切分
                new_x=edge.point_at(0.5).x
                add_cut_mid(new_x,cutX,cutX_mid)                
        x_edges=[LineSeg(Node(x,bounds[0].y),Node(x,bounds[1].y)) for x in cutX_mid]
        y_edges=[LineSeg(Node(bounds[0].x,y),Node(bounds[1].x,y)) for y in cutY_mid]
        for edge in x_edges+y_edges:
            cut_edges+=roof.clips_edge(edge)
            
        res=MergeEdgeAlgo(cut_edges).get_result()
        return res

    def _get_shortest_path(self,vis_graph:dict[Node,list[Node]],source:Node) -> tuple[dict[Node,float],dict[Node,Node]]:
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

