from dataclasses import dataclass
# from dataclasses_json import dataclass_json

# @dataclass_json
@dataclass
class SUWindow:
    partNo: str
    shape: str
    property: str
    materialSupply:str
    enteredBy:str

    type_name:str="Window"
    id_attr:str="partNo"   
