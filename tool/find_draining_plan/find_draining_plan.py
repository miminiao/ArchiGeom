from lib.geom import Node,Loop,Polygon
from lib.biz_algo import BizAlgo
from lib.utils import Constant

class FindDrainingPlan(BizAlgo):
    def __init__(self,roofs:list[Polygon],drain_points:list[Node],const:Constant)->None:
        super().__init__(const=const)
        self.roofs:list[Polygon]=roofs
        self.drain_points:list[Node]=drain_points
        self.roof_drain:dict[Polygon:list[Node]]={}
        """屋面区域->区域包含的落水口"""
        self.good_regions:list[Polygon]=[]
        """布置好排水的区域"""
        self.bad_regions:list[Polygon]=[]
        """没布置好的区域"""
    def _preprocess(self) -> None:
        # 屋面-落水口点位配对
        self.roof_drain={roof:[] for roof in self.roofs}
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
        for roof in self.roofs:
            self.cut_region(roof,self.roof_drain[roof])
        for region in self.bad_regions:
            new_hole=self.find_hole(region)
            if new_hole is not None:
                self.roof_drain[region]=self.roof_drain.get(roof,[])+[new_hole]
                self.cut_region(region,new_hole)
        for region in self.good_regions:
            self.cut_dead_corner(region)
        return self.good_regions,self.bad_regions
    def cut_region(self,region:Polygon,drains:list[Node])->None:
        ...
    def find_hole(self,region:Polygon)->Node:
        ...
    def cut_dead_corner(self,region:Polygon)->None:
        ...

if __name__=="__main__":
    import json
    from tool.converter.json_converter import JsonLoader
    with open("tool/find_drain_plan/test_case.json",'r') as f:
        geoms=json.load(f,object_hook=JsonLoader.from_cad_obj)
    roof_boundaries=[geom for geom in geoms if isinstance(geom,Loop)]
    drain_points=[geom for geom in geoms if isinstance(geom,Node)]