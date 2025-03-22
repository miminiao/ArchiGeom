import importlib.simple
import requests
import json
import asyncio

class Downloader:
    URL="http://192.168.20.113:38405/FileInfo/Download"
    def __init__(self,data_type:type,suffix:str=".dwg"):
        self.data_type=data_type
        self.suffix=suffix
        
        self.type_name=data_type.type_name
        self.id_attr=data_type.id_attr
        module_path="/".join(data_type.__module__.split('.')[:-1])
        self.info_path=f"{module_path}/info.json"
        self.data_path=f"{module_path}/data"
        self.failed_ids_path=f"{module_path}/failed_ids.json"
        self.failed_ids=[]
    def _download_item(self,item_id:str,timeout:float)->None:
        data={
            "fileName":item_id+self.suffix,
            "typeName":self.type_name,
        }
        headers={
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
            "Content-Type":"application/x-www-form-urlencoded",
            "userinfo":"",
        }
        try:
            res=requests.post(url=self.URL,data=data,headers=headers,timeout=timeout)
            with open(f"{self.data_path}/{item_id}{self.suffix}","wb") as f:
                f.write(res.content)
        except requests.exceptions.Timeout:
            self.failed_ids.append(item_id)
    def execute(self,start:int=0,count:int=0,timeout:float=10)->None:
        with open(self.info_path,encoding="utf8") as f:
            info=json.load(f)
        for i,j_obj in enumerate(info[start:],start=start):
            # obj=self.data_type.from_dict(j_obj)
            obj=self.data_type(**j_obj)
            self._download_item(item_id=getattr(obj,self.id_attr),timeout=timeout)
            if i==count-1: break
        with open(self.failed_ids_path,"w") as f:
            json.dump(self.failed_ids,f,ensure_ascii=False)
        print(f"{i+1-len(self.failed_ids)} items downloaded successfully, {len(self.failed_ids)} items failed.")

def get_model(data_type:str):
    match data_type:
        case "building":
            from tool.CADLib.building.model import Building
            return Building
        case "villa":
            from tool.CADLib.villa.model import Villa
            return Villa
        case "kitchen":
            from tool.CADLib.kitchen.model import Kitchen
            return Kitchen
        case "bathroom":
            from tool.CADLib.bathroom.model import Bathroom
            return Bathroom
        case "coretube":
            from tool.CADLib.coretube.model import Coretube
            return Coretube
        case "rental_group":
            from tool.CADLib.rental_group.model import RentalGroup
            return RentalGroup
        case "rental_unit":
            from tool.CADLib.rental_unit.model import RentalUnit
            return RentalUnit
        case "su_railing":
            from tool.CADLib.su_railing.model import SURailing
            return SURailing
        case "su_window":
            from tool.CADLib.su_window.model import SUWindow
            return SUWindow        
        case "su_door":
            from tool.CADLib.su_door.model import SUDoor
            return SUDoor   
        case _:
            raise ValueError("Invalid data type")
    
if __name__=="__main__":
    # 素材库
    downloader=Downloader(data_type=get_model("building"),suffix=".dwg")
    downloader.execute(count=10)
    
    # 厨卫
    # downloader=Downloader(data_type=get_model("kitchen"))
    # for i in range(4):
    #     for j in range(10):
    #         if i==0 and j==0: continue
    #         if i==3 and j==3: break
    #         downloader._download_item(f"THSTD_KICH_{i}{j}",10)
    #     else:continue
    #     break

