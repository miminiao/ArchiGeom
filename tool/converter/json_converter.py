"""Geom<-->json"""

import math
from lib.geom import Geom,Node,LineSeg,Arc,Polyedge,Loop,shPolygon
from lib.linalg import Tensor,Vec3d,Vec4d,Mat3d,Mat4d
from test.CGS.case_model import CGSTestCase

class JsonDumper:
    """Geom类的序列化方法"""
    @staticmethod
    def default(obj:Geom):
        ignore_list=obj._dumper_ignore
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
                return LineSeg(obj["start_point"],obj["end_point"])
            case "arc":
                total_angle=obj["end_angle"]-obj["start_angle"]
                if total_angle<0: total_angle+=math.pi*2
                return Arc(obj["start_point"],obj["end_point"],math.tan(total_angle/4))
            case "polyline":
                pl_nodes=[(seg["start_point"],seg["bulge"]) for seg in obj["segments"]]
                if obj["is_closed"]:
                    # return Loop(pl_nodes)
                    return Loop.from_nodes([Node(*seg["start_point"]) for seg in obj["segments"]])  # TODO 替换Loop的构造方法
                else:
                    return Polyedge(pl_nodes)
            case "hatch":
                ...
            case "text":
                ...
            case None: return obj
            case _: return None
