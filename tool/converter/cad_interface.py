import json
import win32com.client
from pathlib import Path
from tool.converter.cad_object import CADEntity,CADBlockDef
from lib.utils import retry,StopRetry

class CADInterface:
    @retry(interval=0.1)
    def __init__(self, file_path:Path|str=None, version:int=23) -> None:
        """CAD图纸读取接口.

        Args:
            file_path (Paht|str, optional): dwg文件路径. Defaults to None, 读取当前活动文档的选择集内容.
            version (int, optional): CAD版本号. Defaults to 23 (for ACAD2020).
        """
        self._acad = win32com.client.Dispatch(f"AutoCAD.Application.{version}")
        if file_path is None:
            self._doc = self._acad.ActiveDocument
        else:
            path=Path(file_path)
            if not path.exists(): 
                raise StopRetry(FileNotFoundError(f"'{file_path}' does not exist."))
            for doc in self._acad.Documents:
                if Path(doc.FullName)==path.absolute():
                    doc.Activate()
                    self._doc=doc
                    break
            else:
                self._doc=self._acad.Documents.Open(file_path)
            selection = self._doc.ActiveSelectionSet
            selection.Select(5)  # Select All
        self._selection = list(self._doc.ActiveSelectionSet)
        self.doc_name:str=self._doc.Name
        print(f"{self.doc_name}: {len(self._selection)} entities")

    def close(self)->None:
        self._doc.Close(False)

    def get_cad_objects(self,parse_block_def:bool=True)->list[CADEntity|CADBlockDef]:
        CADBlockDef.init_doc_block_table(self._doc.Blocks)
        CADBlockDef.set_parse_flag(parse_block_def)
        print("Parsing..."+" (block def excluded)" if not parse_block_def else "")
        res=[]
        for i,ent in enumerate(self._selection):
            if object_name:=ent.ObjectName!="":
                ...  # DEBUG
            if (parsed_ent:=CADEntity.parse(ent)) is not None:
                res.append(parsed_ent)
        res.extend(CADBlockDef.parsed_blocks.values())
        print("Parsing done.")
        return res
        # cad_geoms=flatten_blocks(cad_objects)
        # geoms=[obj.to_geom() for obj in cad_objects if isinstance(obj,CADEntity)]

if __name__=="__main__":
    
    # from tool.converter.json_converter import JsonDumper
    # dumper=JsonDumper.default
    # dumper=JsonDumper.to_cgs
    dumper=lambda _:_.__dict__
    
    # doc=CADInterface(r"C:\Users\Administrator\Desktop\a.dwg")
    doc=CADInterface()
    cad_objects=doc.get_cad_objects()
    # doc.close()
    with open("./tool/converter/output/case_1.json",'w',encoding="utf8") as f:
        json.dump(cad_objects,f,ensure_ascii=False,default=dumper)
