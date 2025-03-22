"""Geom<-->json"""

import math
from lib.geom import Geom,Node,LineSeg,Arc,Circle,Polyedge,Loop,Polygon
from lib.interval import Interval1d
from lib.linalg import Tensor,Vec3d,Vec4d,Mat3d,Mat4d
from tests.CGS.case_model import CGSTestCase

class JsonDumper:
    """Geom类的序列化方法"""
    @staticmethod
    def default(obj:Geom):
        ignore_list=getattr(obj,"_dumper_ignore",[])
        res={"class_name":obj.__class__.__name__}
        res.update({k:v for k,v in obj.__dict__.items() if k not in ignore_list})
        return res
    @staticmethod
    def to_cgs(obj:Geom|Tensor)->dict|None:
        match obj.__class__.__name__:
            case "Vec3d":
                return {"type":"vector3",**obj.__dict__}
            case "Vec4d":
                return {"type":"vector4",**obj.__dict__}
            case "Mat3d":
                return {"type":"matrix4",
                        "row1":Vec4d(*obj[0]),
                        "row2":Vec4d(*obj[1]),
                        "row3":Vec4d(*obj[2]),
                        "row4":Vec4d.W,}
            case "Mat4d":
                return {"type":"matrix4",
                        "row1":Vec4d(*obj[0]),
                        "row2":Vec4d(*obj[1]),
                        "row3":Vec4d(*obj[2]),
                        "row4":Vec4d(*obj[3]),}
            case "Node":
                return {"type":"point3",**obj.__dict__}
            case "LineSeg":
                return {"type":"line2d",
                        "origin":obj.s,
                        "vector":obj.tangent_at(0) if not obj.is_zero() else Vec3d(1,0,0),  # TODO: fix
                        "minRange":0,
                        "maxRange":obj.length,}
            case _: 
                return obj.__dict__
            
class JsonLoader:
    """Geom类的反序列化方法"""
    @staticmethod
    def from_cgs(obj:dict)->Geom|Tensor:
        if "params" in obj and "expected" in obj:
            return CGSTestCase(obj["params"],obj["expected"])
        if "type" in obj:
            match obj["type"]:
                case "vector3":
                    return Vec3d(obj["x"],obj["y"],obj["z"])
                case "vector4":
                    return Vec4d(obj["x"],obj["y"],obj["z"],obj["w"])
                case "matrix4":
                    return Mat4d.from_row_vecs([obj["row1"],obj["row2"],obj["row3"],obj["row4"]])
                case "point3":
                    return Node(obj["x"],obj["y"],obj["z"])
                case "line2d":
                    origin:Node=obj["origin"]
                    vector:Vec3d=obj["vector"]
                    s=Node.from_vec3d(origin.to_vec3d()+vector*obj["minRange"])
                    e=Node.from_vec3d(origin.to_vec3d()+vector*obj["maxRange"])
                    return LineSeg(s,e)
    @staticmethod
    def from_cad_obj(obj:dict)->Geom:
        match obj.get("object_name",None):
            case "point":
                return Node(*obj["point"])
            case "line":
                s,e=Node(*obj["start_point"]),Node(*obj["end_point"])
                return LineSeg(s,e)
            case "arc":
                total_angle=obj["end_angle"]-obj["start_angle"]
                s,e=Node(*obj["start_point"]),Node(*obj["end_point"])
                if total_angle<0: total_angle+=math.pi*2
                return Arc(s,e,math.tan(total_angle/4))
            case "circle":
                return Circle(obj["center"],obj["radius"])
            case "polyline":
                nodes=[Node(*point) for point in obj["points"]]
                bulges=obj["bulges"]
                if obj["is_closed"]:
                    return Loop(nodes,bulges)
                else:
                    return Polyedge(nodes,bulges)
            case "hatch":
                if len(obj["loops"])==1: 
                    return Polygon(obj["loops"][0])
                shell=max(obj["loops"],key=lambda loop:abs(loop.area))
                holes=[loop for loop in obj["loops"] if loop is not shell]
                return Polygon(shell,holes)
            case "text":
                ...
            case None: return obj
            case _: return None
    @staticmethod
    def default(obj:dict)->Geom:
        match obj.get("class_name",None):
            case "Node":
                return Node(obj["x"],obj["y"],obj["z"])
            case "LineSeg":
                return LineSeg(obj["s"],obj["e"])
            case "Arc":
                return Arc(obj["s"],obj["e"],obj["bulge"])
            case "Circle":
                return Circle(obj["center"],obj["radius"])
            case "Polyedge":
                return Polyedge(obj["nodes"],obj["bulges"])
            case "Loop":
                return Loop(obj["nodes"],obj["bulges"])
            case "Polygon":
                return Polygon(obj["shell"],obj["holes"])
            case "Interval1d":
                return Interval1d(obj["l"],obj["r"],obj["value"])
            case None: return obj
