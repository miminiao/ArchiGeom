import win32com.client
import json
from tool.converter.cad_object import CADEntity,CADBlockDef
def get_cad_objects()->list[CADEntity|CADBlockDef]:
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
            if object_name:=ent.ObjectName!="":
                ...  # DEBUG
            if (parsed_ent:=CADEntity.parse(ent)) is not None:
                if isinstance(parsed_ent,list): 
                    res.extend(parsed_ent)
                else: res.append(parsed_ent)
            i += 1
        except Exception as e:
            if i>last_exception: 
                print(f"Ent {i}: {object_name}",e,"retrying...")
                last_exception=i
    res.extend(CADBlockDef.blocks.values())
    return res

cad_objects=get_cad_objects()

# cad_geoms=flatten_blocks(cad_objects)
# geoms=[obj.to_geom() for obj in cad_objects if isinstance(obj,CADEntity)]

if __name__=="__main__":
    
    # from tool.converter.json_converter import JsonDumper
    # dumper=JsonDumper.default
    # dumper=JsonDumper.to_cgs
    dumper=lambda _:_.__dict__

    with open("./tool/converter/output/case_13.json",'w',encoding="utf8") as f:
        json.dump(cad_objects,f,ensure_ascii=False,default=dumper)

