import json
import win32com.client
from pathlib import Path
from tool.converter.cad_object import CADEntity,CADBlockDef
from lib.utils import retry,StopRetry

class CADInterface:
    """CAD图纸读取接口.

    Args:
        file_path (Path|str, optional): dwg文件路径. Defaults to None (为None时读取当前活动文档的选择集).
        version (int, optional): CAD版本号. Defaults to 23 (for ACAD2020).
        close_doc (bool, optional): 结束时是否关闭当前文档. Defaults to False.
        save_at_close (bool, optional): 关闭时是否保存. close_doc==True时生效. Defaults to False.
    """
    @retry(delay=0.1)
    def __init__(self, file_path:Path|str=None, /,*, 
                 version:int=23,
                 close_at_finish:bool=False,
                 save_at_close:bool=False) -> None:
        self._close_at_finish=close_at_finish
        self._save_at_close=save_at_close
        self._acad = win32com.client.Dispatch(f"AutoCAD.Application.{version}")
        # 未传入文件路径则读取当前选择集, 传入路径则尝试打开并全选
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
        print(f"{self.doc_name}: {len(self._selection)} entities selected.")

    def __enter__(self):
        return self
    
    def __exit__(self,exc_type,exc_val,exc_tb):
        if self._close_at_finish:
            self.close_doc(self._save_at_close)

    def close_doc(self,save:bool=False)->None:
        self._doc.Close(save)

    def get_selected_objects(self,parse_block_def:bool=True)->list[CADEntity|CADBlockDef]:
        CADBlockDef.init_doc_block_table(self._doc.Blocks)
        CADBlockDef.set_parse_flag(parse_block_def)
        print("Parsing..."+(" (block def excluded)" if not parse_block_def else ""))
        res=[]
        for i,ent in enumerate(self._selection):
            parsed_ent=CADEntity.parse(ent)
            if parsed_ent is not None: res.append(parsed_ent)
        res.extend(CADBlockDef.parsed_blocks.values())
        print("Parsing done.")
        return res

if __name__=="__main__":
    # from tool.converter.json_converter import JsonDumper
    # dumper=JsonDumper.default
    # dumper=JsonDumper.to_cgs
    dumper=lambda _:_.__dict__
    
    with CADInterface() as cad:
        cad_objects=cad.get_selected_objects()

    with open("./tool/converter/output/case_1.json",'w',encoding="utf8") as f:
        json.dump(cad_objects,f,ensure_ascii=False,default=dumper)
