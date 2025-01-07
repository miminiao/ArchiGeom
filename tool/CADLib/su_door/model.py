from dataclasses import dataclass
# from dataclasses_json import dataclass_json

# @dataclass_json
@dataclass
class SUDoor:
    partNo: str
    shape: str
    property: str
    materialSupply:str
    enteredBy:str

    type_name:str="Door"
    id_attr:str="partNo"   
