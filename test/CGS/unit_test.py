import json
from lib.geom import Node,LineSeg,Arc,Loop,Polyedge,Polygon
from tool.converter.json_converter import JsonLoader,JsonDumper

f_name="Line2ExtendByDistanceUnit"
with open(f"./test/CGS/unittest_template/{f_name}.json") as f:
    cases=json.load(f,object_hook=JsonLoader.from_cgs)
    for case in cases:
        ## Line2AngelToLine2Unit
        # case.expected=case.params[0].to_vec3d().angle_to(case.params[1].to_vec3d())
        ## Line2ExtendByDistanceUnit
        line=case.params[0]
        vec=line.to_vec3d().unit()
        new_s=Node.from_vec3d(line.s.to_vec3d()-vec*case.params[1])
        new_e=Node.from_vec3d(line.e.to_vec3d()+vec*case.params[2])
        case.expected=LineSeg(new_s,new_e)
...

with open(f"./test/CGS/unittest/{f_name}.json",'w') as f:
    json.dump(cases,f,default=JsonDumper.to_cgs)
...