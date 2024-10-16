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
class RentalUnit:
    floorNo: str|None
    customer: str
    designYear: int
    materialSource: str|None
    familyNo: str
    wideCount: str
    familyWidth: float
    familyDepth:float
    roomNum:float
    bathroomNum:float
    area: float
    familyFeature: str
    downloadCount: int
    viewCount: int
    updatedAt: str

    type_name:str="ModuleUnit"
    id_attr:str="familyNo"   