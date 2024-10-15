import requests as req
import json
from building.model import Family,Building

data_type=Building
type_name="building"
name_attr="floorNo"

unsuccessful_ids=[]
def download(type_name:type,id:str,suffix:str)->None:
    url="http://47.103.58.154:38405/FileInfo/Download"
    data={
        "fileName":id+suffix,
        "typeName":type_name
    }
    try:
        file=req.post(url,data=data,timeout=10)
        with open(f"{type_name}/{id}{suffix}","wb") as f:
            f.write(file.content)
    except req.exceptions.Timeout:
        unsuccessful_ids.append(id)

info_file_path=f"{type_name}/info.json"
with open(info_file_path,encoding="utf8") as f:
    info=json.load(f)

all_data=[]
for j_obj in info:
    obj=data_type.from_dict(j_obj)
    download(type_name=type_name,id=getattr(obj,name_attr),suffix=".dwg")
    if len(all_data)==3: break

with open(f"{type_name}/unsuccessful_ids.json","w"):
    json.dump(unsuccessful_ids,ensure_ascii=False)


