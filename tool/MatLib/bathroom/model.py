from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Bathroom:
    moduleNo: str
    developer: str
    storageYear: int
    layout: str
    setting: str
    widthLower: float
    widthUpper: float
    depthLower: float
    depthUpper: float
    area: float
    characteristic: str
    remark: str|None
    viewCount:int
    downloadCount:int
    updatedAt:str
    
    type_name:str="Bathroom"
    id_attr:str="moduleNo"    
    