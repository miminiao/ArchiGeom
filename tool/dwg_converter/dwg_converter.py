# %%
import win32com.client
import json
from tool.dwg_converter.cad_object import CADEntity,CADBlockDef

if __name__ == "__main__":
    acad = win32com.client.Dispatch("AutoCAD.Application.23")
    # acad = win32com.client.Dispatch("AutoCAD.Application.20")
    doc = acad.ActiveDocument
    msp = doc.ModelSpace
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
            res.append(CADEntity.parse(ent))
            i += 1
        except Exception as e:
            if i>last_exception: 
                print(f"Ent {i}: {object_name}",e,"retrying...")
                last_exception=i

    res.extend(list(CADBlockDef.blocks.values()))

    # %%
    with open("./tool/dwg_converter/output/case_8.1.json",'w',encoding="utf8") as f:
        json.dump(res,f,ensure_ascii=False,default=lambda x:x.__dict__)
