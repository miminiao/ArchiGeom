from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Family:
    familyNo: str
    wideCount: str
    familyWidth: float
    familyDepth:float
    roomNum:float
    bathroomNum:float
    area: float
    familyFeature: str

@dataclass_json
@dataclass
class RentalGroup:
    floorNo: str
    customer: str
    designYear: int
    materialSource: str|None
    buildingLayout: str
    floorArea: float
    width: float
    depth: float
    roomRate: float
    downloadCount: int
    viewCount: int
    updatedAt: str
    families: list[Family]

    type_name:str="GroupPlane"
    id_attr:str="floorNo"    
    