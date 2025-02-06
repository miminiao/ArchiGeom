from dataclasses import dataclass
# from dataclasses_json import dataclass_json

# @dataclass_json
@dataclass
class SURailing:
    partNo: str
    materialType: str
    handRailProperty: str
    materialSource:str
    enteredBy:str

    type_name:str="HandRail"
    id_attr:str="partNo"   
