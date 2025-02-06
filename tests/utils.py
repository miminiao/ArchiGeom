import json
from lib.geom import Geom
from tool.converter.json_converter import JsonLoader,JsonDumper

_ROOT_DIR=""
def set_root_dir(root_dir:str)->None:
    global _ROOT_DIR
    _ROOT_DIR=root_dir

def read_case(sub_test,file_name,hook_mode=None)->list[Geom]:
    if hook_mode=="cad": hook=JsonLoader.from_cad_obj
    else: hook=JsonLoader.default
    path=_ROOT_DIR+sub_test[0]+file_name+".json"
    with open(path,'r') as f:
        objs=json.load(f,object_hook=hook)
    return objs

def write_stdout(res:list[Geom],sub_test,file_name)->None:
    path=_ROOT_DIR+sub_test[0]+file_name+".json"
    with open(path,'w') as f:
        json.dump(res,f,default=JsonDumper.default)