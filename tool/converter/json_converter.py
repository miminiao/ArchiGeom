from lib.geom import Geom,Node,LineSeg,Arc,Polyline,Loop,Polygon
from lib.linalg import Tensor,Vec3d,Vec4d,Mat3d,Mat4d
from test.CGS.case_model import CGSTestCase

class JsonDumper:
    default=lambda _:_.__dict__
    @classmethod
    def to_cgs(cls,obj:Geom|Tensor)->dict|None:
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
    @classmethod
    def from_cgs(cls,obj:dict)->Geom|Tensor:
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
    @classmethod
    def from_cad_obj(cls,obj:dict)->Geom:
        match obj["object_name"]:
            case "point":
                return Node(obj)
            case "line":
                return Node.from_cad_obj(obj)
            case "arc":
                ...
            case "polyline":
                ...
            case "hatch":
                ...
            case "text":
                ...
                
            case _: return None
def cad_polyline_to_loop(j_obj:list)->list[Loop]:
    loops=[]
    nodes=[]
    for obj in j_obj:
        if obj["object_name"]=="polyline":
            edges=[]
            seg_num=len(obj["segments"]) if obj["is_closed"] else len(obj["segments"])-1
            for i in range(seg_num):
                seg=obj["segments"][i]
                next_seg=obj["segments"][(i+1)%len(obj["segments"])]
                x1,y1,_=seg["start_point"]
                x2,y2,_=next_seg["start_point"]
                lw=rw=seg["start_width"]/2
                bulge=seg["bulge"]
                s=Node(x1,y1)
                e=Node(x2,y2)
                if s.equals(e):continue
                s=Node.find_or_insert_node(s,nodes,copy=True)
                e=Node.find_or_insert_node(e,nodes,copy=True)
                if abs(bulge)<Arc.const.TOL_VAL:
                    edges.append(LineSeg(s,e))
                else:
                    edges.append(Arc(s,e,bulge))
            loops.append(Loop(edges))
    return loops