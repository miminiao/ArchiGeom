from dataclasses import dataclass
# from dataclasses_json import dataclass_json

# @dataclass_json
@dataclass
class Family:
    familyNo: str
    wideCount: float
    dwellingFeature: str
    area: float
    airconditionForm: str
    familyFeature: str

# @dataclass_json
@dataclass
class Building:
    floorNo: str
    customer: str
    province: str
    city: str
    designYear: int
    materialSource: str|None
    residentType: str
    applicableHeight: str
    houseHold: str
    floorArea: float
    width: float
    depth: float
    roomRate: float
    stairsCount: int
    elevatorCount: int
    coreTubeInfo: str
    floorFeature: str
    downloadCount: int
    viewCount: int
    updatedAt: str
    families: list[Family]

    type_name:str="Building"
    id_attr:str="floorNo"    
