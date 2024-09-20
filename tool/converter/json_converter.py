from lib.geom import Node,LineSeg,Arc,Polyline,Loop,Polygon
from lib.linalg import Vec3d,Mat3d,Mat4d

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

class JsonDumper:
    default=lambda _:_.__dict__
    @classmethod
    def to_cgs(cls,obj)->dict|None:
        match obj.__class__.__name__:
            case "Vec3d":
                return {"type":"vector3"}.update(obj.__dict__)
            case "Vec4d":
                return {"type":"vector4"}.update(obj.__dict__)
            case "Mat4d":
                return {"type":"matrix4","row1":obj[0],"row2":obj[1],"row3":obj[2],"row4":obj[3],}
