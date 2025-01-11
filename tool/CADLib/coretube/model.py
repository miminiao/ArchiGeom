from dataclasses import dataclass

@dataclass
class Coretube:
    moduleNo: str
    applicableRegion: str
    storageYear: int
    applicableHeight: str
    houseHold: str
    floorFeature: str
    coreTubeForm: str
    frontRoom: str
    floorForm: str
    stairsCount: int
    elevatorCount:int
    courtYard:str
    area:float
    coreTubeFeature:str
    remark:str|None
    downloadCount: int
    viewCount: int
    updatedAt: str

    type_name:str="Coretube"
    id_attr:str="moduleNo"
    