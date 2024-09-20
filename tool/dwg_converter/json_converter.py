from lib.geom import Geom,Node,LineSeg,Arc,Loop
class CADJsonLoader:
    @staticmethod
    def polyline_to_loop(j_obj:list)->list[Loop]:
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
    def CGS_dumper(cls,geom:Geom|)->dict:
        match geom.__class__.__name__:
            case "Node":
                return {"type":"point3","x":geom.x,"y":geom.y,"z":geom.z}
            case "LineSeg":
                return {"type":"line2","start":JsonDumper.CGS_dumper(geom.s),"end":JsonDumper.CGS_dumper(geom.e)}
