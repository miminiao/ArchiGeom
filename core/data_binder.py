#%%
ROOT_PATH="..\\"
DATA_PATH=ROOT_PATH+"data\\"

import os
import json
import modelling_data_constructor as mdc
import analytical_data_constructor as adc

#%% 从dwgData和ToACAEntity/ToACABlock/ToACAFloor/Slab载入
def findLatestFile(prefix):
    for root, ds, fs in os.walk(DATA_PATH):
        name=""
        for f in fs:
            if f.endswith('.json') and f.startswith(prefix) and f>name:
                name=f
        return DATA_PATH+name
with open(findLatestFile("dwgData"), "rb") as f:
    jsonData=json.load(f)
with open(findLatestFile("ToACAEntity"),"rb") as f:
    jsonEntities=json.load(f)
with open(findLatestFile("ToACAFloor"),"rb") as f:
    jsonFloors=json.load(f)
with open(findLatestFile("ToACABlock"),"rb") as f:
    jsonBlocks=json.load(f)    
with open(findLatestFile("Slab"),"rb") as f:
    jsonSlabs=json.load(f)
with open(findLatestFile("ACASlab"),"rb") as f:
    jsonACASlab=json.load(f)

dataBuilding=adc.Building(jsonData,jsonACASlab)
modelBuilding=mdc.Building(jsonFloors,jsonBlocks["_hash"],jsonEntities["_hash"],jsonSlabs)

#%% 从已拼装的OutputModel和OutputData载入
# with open("D:\Desktop\结构数据对接\建筑数据\样例\OutputModel.json","rb") as f:
#     modelBuilding=mdc.Building.init_fromOutput((json.load(f)))
# with open("D:\Desktop\结构数据对接\建筑数据\样例\OutputData.json","rb") as f:
#     dataBuilding=adc.Building(json.load(f))

#%% output
def toDict(obj):
    ignoreList=["parent","child"]
    d=obj.__dict__.copy()
    for attr in ignoreList:
        if attr in obj.__dict__:
            del(d[attr])
    return d
if __name__=="__main__":
    with open("OutputModel.json","w",encoding="utf-8") as f:
        json.dump(modelBuilding,f,ensure_ascii=False,default=lambda o:o.__dict__)
    with open("OutputData.json","w",encoding="utf-8") as f:
        json.dump(dataBuilding,f,ensure_ascii=False,default=toDict)
    with open(DATA_PATH+"Slab.json","w",encoding="utf-8") as f:
        json.dump(modelBuilding.Slabs,f,ensure_ascii=False,default=toDict) 



