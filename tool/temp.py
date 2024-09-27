import json
from tool.converter.json_converter import JsonLoader,JsonDumper

f_name="Line2SetRangeUnit_int"
with open(f"./test/CGS/cases/{f_name}.json") as f:
    cases=json.load(f,object_hook=JsonLoader.from_cgs)
with open(f"./test/CGS/cases/{f_name}_int.json",'w') as f:
    json.dump(cases,f,default=JsonDumper.to_cgs)
...