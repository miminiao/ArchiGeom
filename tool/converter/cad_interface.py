import win32com.client
import json
from tool.converter.cad_object import CADEntity,CADBlockDef

acad = win32com.client.Dispatch("AutoCAD.Application.23")
# acad = win32com.client.Dispatch("AutoCAD.Application.20")
doc = acad.ActiveDocument
sel = doc.ActiveSelectionSet
sel_num = len(sel)
print(f"{doc.Name}: {sel_num} entities")

CADBlockDef.init_doc_block_table(doc.Blocks)
res=[]
i = 0
last_exception=-1
while i < sel_num:
    try:
        ent = sel[i] 
        if object_name:=ent.ObjectName=="":
            ...  # DEBUG
        if (parsed_ent:=CADEntity.parse(ent)) is not None:
            res.append(parsed_ent)
        i += 1
    except Exception as e:
        if i>last_exception: 
            print(f"Ent {i}: {object_name}",e,"retrying...")
            last_exception=i

res.extend(CADBlockDef.blocks.values())

import tool.converter.json_converter as jconv
# dumper=jconv.JsonDumper.default
dumper=jconv.JsonDumper.to_cgs
res=[ent.to_geom() for ent in res if isinstance(ent,CADEntity)]
with open("./tool/converter/output/output.json",'w',encoding="utf8") as f:
    json.dump(res,f,ensure_ascii=False,default=dumper)