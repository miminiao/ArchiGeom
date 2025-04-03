import math
from dataclasses import dataclass
from lib.geom import Node,LineSeg,Loop,Polygon
from lib.linalg import Vec3d
from lib.geom_algo import BooleanOperation, FindLoopAlgo, MergeEdgeAlgo
from lib.biz_algo import BizAlgo
from lib.geom_plotter import CADPlotter
from lib.utils import Timer

import shapely

INF=math.inf
Timer.enable()

@dataclass
class DrainingRegion:
    poly: Polygon
    target_drain: Node=None
    direction: Vec3d=None
    dist:float=math.inf
    sub_regions: list["DrainingRegion"]=None

class FindDrainingPlan(BizAlgo):
    def __init__(self,roofs:list[Polygon],drain_points:list[Node],cut_tol:float=600)->None:
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
    def get_result(self):
        self._preprocess()
        roof_drain=self._match_roof_drain(self.roofs,self.drain_points)
        safe_roofs=[]
        uncertain_roofs=[]
        bad_roofs=[]
        for roof,drains in roof_drain.items():
            match len(drains):
                case 0: uncertain_roofs.append(roof)
                case 1: safe_roofs.append(DrainingRegion(roof,drains[0]))
                case _: safe_roofs+=self._cut_roof(roof,drains)

        CADPlotter.draw_geoms([r.poly for r in safe_roofs])
        with open('log.csv','w') as f:
            f.write('\n'.join([log[0]+' '+str(log[1]) for log in Timer.logs]))
        exit()

        for reg in safe_roofs: 
            self._get_converge_edges(reg,uncertain_roofs)
        for reg in uncertain_roofs:
            self._find_tunnel(reg,safe_roofs)
            self._find_converge_edges(reg,bad_roofs)
        self._postprocess()
        return safe_roofs,bad_roofs

    def _match_roof_drain(self,roofs:list[Polygon],drain_points:list[Node])->dict[Polygon,list[Node]]:
        """屋面区域-落水口点位配对"""
        roof_drain:dict[Polygon,list[Node]]={roof:[] for roof in roofs}
        for drain_point in drain_points:
            min_dist=float("inf")
            for roof in roofs:
                if roof.covers(drain_point):
                    roof_drain[roof].append(drain_point)
                    break
                pt=roof.closest_point(drain_point)
                if dist:=drain_point.dist(pt)<min_dist:
                    min_dist,closest_pt=dist,pt
            else:
                roof_drain[roof].append(closest_pt)
        return roof_drain
    @Timer
    def _cut_roof(self,roof:Polygon,drains:list[Node])->list[DrainingRegion]:
        """将屋面切分为若干汇水区域，和落水口一一对应"""
        # 划分单元格
        roof_edges=sum([list(loop.edges) for loop in roof.all_loops],[])
        candidate_edges=self._get_candidate_edges(roof,drains)
        dual_edges=[edge.opposite() for edge in candidate_edges]
        cell_loops=FindLoopAlgo(roof_edges+candidate_edges+dual_edges,directed=True).get_result()

        # 单元格中心点作为代表
        center_points=[loop.get_centroid() for loop in cell_loops]
        cells={c:DrainingRegion(Polygon(loop)) for c,loop in zip(center_points,cell_loops)}

        # 构建可见性图
        vis_graph=self._build_vis_graph(roof,center_points,drains)
        
        # 计算每个单元格区域的落水口
        self._update_target_of_cells(drains,cells,vis_graph)

        # 沿着分水线切屋面
        return self._find_sub_roofs(
            roof=roof,
            drains=drains,
            candidate_edges=candidate_edges, 
            cells=cells, 
            vis_graph=vis_graph,
        )
            
    def _get_candidate_edges(self,roof:Polygon,drains:list[Node])->list[LineSeg]:
        """将屋面横竖划分为若干子区域；划分位置：边界顶点、落水口平分线"""
        bounds=roof.get_aabb()
        div_edges=[]
        # TODO 切的时候，减少近距离的平行线：寻找射线两侧距离接近的点，依次连接
        # 1. 用区域顶点切分
        cutX,cutY=[],[]
        for loop in roof.all_loops:
            for i,node in enumerate(loop.nodes):
                if loop.edge(i-1).tangent_at(0).angle_between(loop.edge(i).tangent_at(0))<self.const.TOL_ANG: continue
                x_line=LineSeg(Node(node.x,bounds[0].y),Node(node.x,bounds[1].y))
                x_segs=roof.clips_edge(x_line)
                for seg in x_segs:
                    if seg.s==node or seg.e==node:
                        div_edges.append(seg)
                        cutX.append(node.x)
                y_line=LineSeg(Node(bounds[0].x,node.y),Node(bounds[1].x,node.y))
                y_segs=roof.clips_edge(y_line)
                for seg in y_segs:
                    if seg.s==node or seg.e==node:
                        div_edges.append(seg)
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
            div_edges+=roof.clips_edge(edge)
            
        res=MergeEdgeAlgo(div_edges).get_result()
        return res

    def _triangulate_drains(self,drains:list[Node])->list[LineSeg]:
        if len(drains)<=1: return []
        if len(drains)==2: return [LineSeg(*drains)]
        points=shapely.MultiPoint([[node.x,node.y] for node in drains])
        lines=shapely.delaunay_triangles(points,tolerance=self.const.TOL_DIST,only_edges=True)
        edges=[LineSeg(Node(*line.coords[0]),Node(*line.coords[1])) for line in lines.geoms]
        merged=MergeEdgeAlgo(edges,break_at_intersections=True).get_result()
        return merged
    @Timer
    def _build_vis_graph(self,roof:Polygon,reg_centers:list[Node],drains:list[Node])->dict[Node,set[Node]]:  # TODO: 效率优化
        """计算房间内可见性图"""
        roof_nodes=sum([loop.nodes for loop in roof.all_loops],[])
        vis_graph={node:set() for node in roof_nodes+reg_centers+drains}  # 可见性图（邻接表）
        # 加入可直达的边
        for i in roof_nodes+drains:
            for j in roof_nodes+reg_centers:
                vis_line=LineSeg(i,j)
                if roof.covers(vis_line):
                    vis_graph[i].add(j)
        return vis_graph    

    def _update_target_of_cells(self, drains, regions, vis_graph):
        for reg in regions.values():
            if reg.target_drain in drains:
                reg.target_drain=None
                reg.dist=INF
        for drain in drains:
            dist_to,pre=self._get_shortest_path(vis_graph,drain)
            for node in regions:
                if dist_to[node]<regions[node].dist:
                    regions[node].dist=dist_to[node]
                    # path=[node]
                    # while path[-1] is not drain:
                    #     path.append(pre[path[-1]])
                    # regions[node].path=Polyedge(path)
                    regions[node].target_drain=drain

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
    @Timer
    def _find_sub_roofs(self,roof:Polygon, drains:list[Node], candidate_edges:list[LineSeg], cells:dict[Node,DrainingRegion], vis_graph:dict[Node,set[Node]]):
        """沿着分水线切屋面"""

        if len(drains)<2:
            return [DrainingRegion(roof,drains[0],sub_regions=cells.values())]
        
        # 枚举分水线，计算切完之后的loss，取最小的方案
        min_loss=INF
        for edge in candidate_edges:
            removed=self._prune_vis_graph(vis_graph,[edge])
            self._update_target_of_cells(drains,cells,vis_graph)
            loss=self._get_total_loss(cells,edge)
            if loss<min_loss:
                min_loss=loss
                new_div_edges=[edge]
            self._recover_vis_graph(vis_graph,removed)
        
        # 用新的div edges切屋面
        self._prune_vis_graph(vis_graph,new_div_edges)
        self._update_target_of_cells(drains,cells,vis_graph)
        dual_edges=[edge.opposite() for edge in new_div_edges]
        roof_edges=sum([list(loop.edges) for loop in roof.all_loops],[])
        new_loops=FindLoopAlgo(roof_edges+new_div_edges+dual_edges,directed=True).get_result()            
        new_roofs=BooleanOperation._loop2polygon(new_loops)
        new_roof_drain=self._match_roof_drain(new_roofs,drains)
        sub_roofs=[]
        for new_roof,new_drains in new_roof_drain.items():
            new_candidate_edges=sum([new_roof.clips_edge(edge) for edge in candidate_edges],[])
            new_cells={center:region for center,region in cells.items() if new_roof.contains(center)}
            sub_roofs+=self._find_sub_roofs(
                roof=new_roof, 
                drains=new_drains,  
                candidate_edges=new_candidate_edges, 
                cells=new_cells, 
                vis_graph=vis_graph,
            )
        return sub_roofs
    
    def _get_total_loss(self,cells:dict[Node,DrainingRegion],edge:LineSeg)->float:
        base_loss=sum([reg.dist*reg.poly.area for reg in cells.values()])  # 落水口距离*区域面积加权
        loss_factor_1=1.0  # 分水线长度短的，加分
        roof_points=sum([loop.nodes for roof in self.roofs for loop in roof.all_loops],[])
        loss_factor_2=1.0  # 当分水线从屋面顶点出发时，加分
        for p in roof_points:
            if p==edge.s or p==edge.e:
                loss_factor_2-=0.0
                break
        return base_loss*loss_factor_1*loss_factor_2
    @Timer
    def _prune_vis_graph(self,vis_graph:dict[Node,set[Node]],new_div_edges:list[LineSeg])->list[list[Node]]:
        removed=[]
        for i in vis_graph:
            for j in vis_graph[i]:
                for edge in new_div_edges:
                    inter=LineSeg(i,j).intersection(edge)
                    if len(inter)>0:  # 被切到的变为不可见
                        removed.append([i,j])
                        break
        for i,j in removed:
            vis_graph[i].remove(j)
        return removed

    def _recover_vis_graph(self,vis_graph:dict[Node,set[Node]],removed:list[list[Node]])->None:
        for i,j in removed:
            vis_graph[i].add(j)

    def _get_converge_edges(self,roof:DrainingRegion,uncertain:list[DrainingRegion])->list[LineSeg]:
        safe_slopes=[]
        converge_edges:list[LineSeg]=[]
        bounds=roof.poly.get_aabb()
        w,h=bounds[1].x-bounds[0].x,bounds[1].y-bounds[0].y
        vecs=[Vec3d(1,1),Vec3d(1,-1),Vec3d(-1,1),Vec3d(-1,-1)]
        crossing_rays=[LineSeg.from_origin_direction_length(roof.target_drain,vec,w+h) for vec in vecs]
        for ray in crossing_rays:
            clips=roof.poly.clips_edge(ray)
            converge_edges+=[edge for edge in clips if edge.s==roof.target_drain or edge.e==roof.target_drain]
        dual_edges=[edge.opposite() for edge in converge_edges]
        roof_edges=sum([list(loop.edges) for loop in roof.poly.all_loops],[])
        new_loops=FindLoopAlgo(roof_edges+converge_edges+dual_edges,directed=True).get_result()            
        new_slopes=BooleanOperation._loop2polygon(new_loops) 
        for slope in new_slopes:
            direction=self._get_direction(slope,roof.target_drain)
            projected_region=self._project_converge_edges(slope,roof.target_drain,direction)
            remained_regions=BooleanOperation.difference(slope,projected_region)
            if len(remained_regions)==0:
                safe_slopes.append(DrainingRegion(slope,roof.target_drain,direction=direction))
            else:
                for reg in remained_regions:
                    new_virtual_drain=[]
                    for edge1 in reg.edges:
                        for edge2 in projected_region.shell.edges:
                            if len(edge1.overlap(edge2))>0:
                                new_virtual_drain.append(edge1.e if edge1.to_vec3d().dot(direction)>0 else edge1.s)
                                break
                        else: continue
                        break
                    if len(new_virtual_drain)==1:
                        safe_slopes+=self._get_converge_edges(DrainingRegion(reg,new_virtual_drain[0]),uncertain)
                    else:
                        sub_regions=self._cut_roof(reg,new_virtual_drain)
                        safe_slopes+=self._get_converge_edges(sub_regions,uncertain)
        return safe_slopes
    def _get_direction(self,slope:Polygon,drain:Node)->Vec3d:
        ...  # TODO
    def _project_converge_edges(self,slope:Polygon,drain:Node,direction:Vec3d)->Polygon:
        ...  # TODO

if __name__=="__main__":
    import json
    from tool.converter.json_converter import JsonLoader
    with open("apps/find_draining_plan/case_2.json",'r') as f:
        geoms=json.load(f,object_hook=JsonLoader.from_cad_obj)
    roof_boundaries=[geom for geom in geoms if isinstance(geom,Loop)]
    drain_points=[geom for geom in geoms if isinstance(geom,Node)]
    roofs=BooleanOperation._rebuild_loop_topology(roof_boundaries)
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

