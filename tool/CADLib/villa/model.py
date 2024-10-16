from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Family:
    familyNo: str
    upperGroundNum: str
    lowerGroundNum: str
    wideCount: float
    roomNum:float
    bathroomNum:float
    area: float
    indoorElevator:str|None
    freeAreaForm:str
    familyFeature: str

@dataclass_json
@dataclass
class Villa:
    floorNo: str
    customer: str
    province: str
    city: str
    designYear: int
    materialSource: str|None
    splicingForm: str
    upperGroundNum: str
    lowerGroundNum: str
    width: float
    depth: float
    floorForm: str
    downloadCount: int
    viewCount: int
    updatedAt: str
    families: list[Family]

    type_name:str="villa"
    id_attr:str="floorNo"    
    