import uuid

class SUGroup:
    pass
class XRefStyle:
    pass
class Building:
    #默认从4个ToACA.json创建
    def __init__(self,jsonFloors,jsonBlocks,jsonEntities,jsonSlabs):
        #构造标准层对象
        self.Blocks={}
        for key in jsonBlocks:
            self.Blocks[key]=Block(jsonBlocks[key])
            for handle in jsonBlocks[key]["SU_EntityInBlock"]:
                try:
                    self.Blocks[key].add(jsonEntities[str(handle)])
                except KeyError as e:
                    print("Missing Entity in Block ",key,e)
        #构造楼层对象
        self.Floors=[]
        if "floorInfos" in jsonFloors:
            for floor in jsonFloors["floorInfos"]:
                newFloor=Floor(floor)
                self.Floors.append(newFloor)
        #构造楼板对象
        self.Slabs=[]
        for slab in jsonSlabs:
            self.Slabs.append(Slab(slab))
        #项目信息
        #self.project_status=jsonFloors["project_status"]
        if "project_name" in jsonFloors:
            self.project_name=jsonFloors["project_name"]
        #self.project_constructor=jsonFloors["project_constructor"]
        #self.project_city=jsonFloors["project_city"]
        if "project_Number" in jsonFloors:
            self.project_Number=jsonFloors["project_Number"]

    #也可以从已经输出过的OutputModel创建
    @classmethod
    def init_fromOutput(cls,jsonOutput):
        newBuilding=cls({},{},{},{})
        for key in jsonOutput["Blocks"]:
            newBuilding.Blocks[key]=Block.init_fromOutput((jsonOutput["Blocks"][key]))
        for floor in jsonOutput["Floors"]:
            newBuilding.Floors.append(Floor(floor))
        for slab in jsonOutput["Slabs"]:
            newBuilding.Slabs.append(Slab(slab))        
        newBuilding.project_name=jsonOutput["project_name"]
        newBuilding.project_Number=jsonOutput["project_Number"]
        return newBuilding
    
    
    def getWalls(self):
        Walls=[]
        for block in self.Blocks.values():
            Walls=Walls+block.Walls
        return Walls
    def getWindows(self):
        Windows=[]
        for block in self.Blocks.values():
            Windows=Windows+block.Windows
        return Windows
    def getDoors(self):
        Doors=[]
        for block in self.Blocks.values():
            Doors=Doors+block.Doors
        return Doors            
    def getRailings(self):
        Railings=[]
        for block in self.Blocks.values():
            Railings=Railings+block.Railings
        return Railings
    def getMassElements(self):
        MassElements=[]
        for block in self.Blocks.values():
            MassElements=MassElements+block.MassElements
        return MassElements
    def getWindow_MassElements(self):
        Window_MassElements=[]
        for block in self.Blocks.values():
            Window_MassElements=Window_MassElements+block.Window_MassElements
        return Window_MassElements
    def getDoor_MassElements(self):
        Door_MassElements=[]
        for block in self.Blocks.values():
            Door_MassElements=Door_MassElements+block.Door_MassElements
        return Door_MassElements
    def getStairs(self):
        Stairs=[]
        for block in self.Blocks.values():
            Stairs=Stairs+block.Stairs
        return Stairs
class Floor:
    def __init__(self,dictArgs):
        self.floor_no=dictArgs["floor_no"]
        self.floor_name=dictArgs["floor_name"]
        #self.aca_src_floor_name=dictArgs["aca_src_floor_name"]
        self.floor_serial=dictArgs["floor_serial"]
        self.floor_elevation=dictArgs["floor_elevation"]
        self.floor_height=dictArgs["floor_height"]
        #self.floor_state=dictArgs["floor_state"]
        self.basePoint=dictArgs["basePoint"]

class Block:
    def __init__(self,dictArgs):
        if "Name" in dictArgs:
            self.Name=dictArgs["Name"]
        else: 
            self.Name=None
        self.Walls=[]
        self.Windows=[]
        self.Doors=[]
        self.Railings=[]
        self.MassElements=[]
        self.Window_MassElements=[]
        self.Door_MassElements=[]
        self.Stairs=[]
#        self.add(dictArgs)
    @classmethod
    def init_fromOutput(cls,jsonOutput):
        newBlock=cls({})
        newBlock.Name=jsonOutput["Name"]
        for item in jsonOutput["Walls"]:
            newBlock.add(item)
        for item in jsonOutput["Windows"]:
            newBlock.add(item)
        for item in jsonOutput["Doors"]:
            newBlock.add(item)
        for item in jsonOutput["Raillings"]:
            newBlock.add(item)
        for item in jsonOutput["MassElements"]:
            newBlock.add(item)
        for item in jsonOutput["Window_MassElements"]:
            newBlock.add(item)
        for item in jsonOutput["Door_MassElements"]:
            newBlock.add(item)
        for item in jsonOutput["Stairs"]:
            newBlock.add(item)
        return newBlock
    def add(self,entity):
        if entity["Type"]=="Wall":
            self.addWall(entity)
        elif entity["Type"]=="window":
            self.addWindow(entity)
        elif entity["Type"]=="door":
            self.addDoor(entity)
        elif entity["Type"]=="Railing":
            self.addRailing(entity)
        elif entity["Type"]=="MassElement":
            self.addMassElement(entity)
        elif entity["Type"]=="window_MassElement":
            self.addWindow_MassElement(entity)
        elif entity["Type"]=="Door_MassElement":
            self.addDoor_MassElement(entity)
        elif entity["Type"]=="Stair":
            self.addStair(entity)
    def addWall(self,dictArgs):
        self.Walls.append(Wall(dictArgs))
    def addDoor(self,dictArgs):
        self.Doors.append(Door(dictArgs))
    def addWindow(self,dictArgs):
        self.Windows.append(Window(dictArgs))
    def addRailing(self,dictArgs):
        self.Railings.append(Railing(dictArgs))
    def addMassElement(self,dictArgs):
        self.MassElements.append(MassElement(dictArgs))
    def addWindow_MassElement(self,dictArgs):
        self.Window_MassElements.append(Window_MassElement(dictArgs))
    def addSlab(self,dictArgs):
        self.Slabs.append(Slab(dictArgs))
    def addDoor_MassElement(self,dictArgs):
        self.Door_MassElements.append(Door_MassElement(dictArgs))
    def addStair(self,dictArgs):
        self.Stairs.append(Stair(dictArgs))
class Wall:
    def __init__(self,dictArgs):
        self.StyleName=dictArgs["StyleName"]
        self.Width=dictArgs["Width"]
        self.Height=dictArgs["Height"]
        self.Alignment=dictArgs["Alignment"]
        #self.Elevation=dictArgs["Elevation"]
        self.StartPoint=dictArgs["StartPoint"]
        self.EndPoint=dictArgs["EndPoint"]
        self.leftOffset=dictArgs["leftOffset"]
        self.rightOffset=dictArgs["rightOffset"]
        self.Type=dictArgs["Type"]
        self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        #--------------------------新增-------------------------------
        self.Material=None #外观材质
        try:
            self.Material=dictArgs["Material"]
        except KeyError as e:
            print("mdc.Wall ",self.Handle,"MissingKey",e)  
class Door:
    def __init__(self,dictArgs):
        self.StartPoint=dictArgs["StartPoint"]
        self.EndPoint=dictArgs["EndPoint"]
        self.StyleName=dictArgs["StyleName"]
        self.Width=dictArgs["Width"]
        self.Height=dictArgs["Height"]
        self.Distance=dictArgs["Distance"]
        self.FlipX=dictArgs["FlipX"]
        self.FlipY=dictArgs["FlipY"]
        self.OwnerHandle=dictArgs["OwnerHandle"]
        self.SU_OwnerID=dictArgs["SU_OwnerID"]
        self.Type=dictArgs["Type"]
        self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        #--------------------------新增-------------------------------
        self.Door_MassElement_style_name=None #门体量/雨篷样式
        self.ACAMassElemID=None
        self.Attributes=None #附加属性
        self.DWGRef=None #大样图文件
        self.ElevationBlockName=None #立面图块
        self.ElevationAttributes=None #立面附加属性
        self.SectionBlockName=None #断面图块
        self.SectionAttributes=None #断面附加属性
        self.Material=None #外观材质
        try:
            self.ACAMassElemID=dictArgs["ACAMassElemID"]
            self.Attributes=dictArgs["Attributes"]
            self.Door_MassElement_style_name=dictArgs["Door_MassElement_style_name"]
            self.DWGRef=dictArgs["DWGRef"]
            self.ElevationBlockName=dictArgs["ElevationBlockName"]
            self.ElevationAttributes=dictArgs["ElevationAttributes"]
            self.SectionBlockName=dictArgs["SectionBlockName"]
            self.SectionAttributes=dictArgs["SectionAttributes"]
            self.Material=dictArgs["Material"]
        except KeyError as e:
            print("mdc.Door ",self.Handle,"MissingKey",e)
class Window:
    def __init__(self,dictArgs):
        self.Width2=dictArgs["Width2"]
        self.StartPoint=dictArgs["StartPoint"]
        self.EndPoint=dictArgs["EndPoint"]
        self.SillHeight=dictArgs["SillHeight"]
        self.CornerFlag=dictArgs["CornerFlag"]
        self.Windows_wall_style_name=dictArgs["Windows_wall_style_name"]
        self.ACAMassElemID=dictArgs["ACAMassElemID"]
        self.StyleName=dictArgs["StyleName"]
        self.Width=dictArgs["Width"]
        self.Height=dictArgs["Height"]
        self.Distance=dictArgs["Distance"]
        self.FlipX=dictArgs["FlipX"]
        self.FlipY=dictArgs["FlipY"]
        self.OwnerHandle=dictArgs["OwnerHandle"]
        self.SU_OwnerID=dictArgs["SU_OwnerID"]
        self.Type=dictArgs["Type"]
        self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        #--------------------------新增-------------------------------
        self.Attributes=None
        self.DWGRef=None #大样图文件
        self.ElevationBlockName=None #立面图块
        self.ElevationAttributes=None #立面附加属性
        self.SectionBlockName=None #断面图块
        self.SectionAttributes=None #断面附加属性
        self.Material=None #外观材质:Material
        try:
            self.Attributes=dictArgs["Attributes"]
            self.DWGRef=dictArgs["DWGRef"]
            self.ElevationBlockName=dictArgs["ElevationBlockName"]
            self.ElevationAttributes=dictArgs["ElevationAttributes"]
            self.SectionBlockName=dictArgs["SectionBlockName"]
            self.SectionAttributes=dictArgs["SectionAttributes"]
            self.Material=dictArgs["Material"]         
        except KeyError as e:
            print("mdc.Window ",self.Handle,"MissingKey",e)
class Railing:
    def __init__(self,dictArgs):
        self.StyleName=dictArgs["StyleName"]
        self.Height=dictArgs["Height"]
        self.Elevation=dictArgs["Elevation"]
        self.StartPoint=dictArgs["StartPoint"]
        self.EndPoint=dictArgs["EndPoint"]
        self.Type=dictArgs["Type"]
        self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        #--------------------------新增-------------------------------
        self.Attributes=None #附加属性:{string:Value}
        self.DWGRef=None #大样图文件:string
        self.ElevationBlockName=None #立面图块:string
        self.ElevationAttributes=None #立面附加属性:{string:Attribute}
        self.UnitBlockName=None #立面单元图块:string
        self.UnitDivided=None #立面单元划分：{类型str="Equally"/"Fixed":最大间隔float}
        self.UnitFilled=None #立面单元填充：str="Full"/"Fixed"
        self.UnitAttributes=None #立面单元附加属性:{string:Attribute}        
        self.SectionBlockName=None #断面图块:string
        self.SectionAttributes=None #断面附加属性:{string:Attribute}
        self.Material=None #外观材质:Material
        try:
            self.Attributes=dictArgs["Attributes"]
            self.DWGRef=dictArgs["DWGRef"]
            self.ElevationBlockName=dictArgs["ElevationBlockName"]
            self.ElevationAttributes=dictArgs["ElevationAttributes"]
            self.UnitBlockName=dictArgs["UnitBlockName"]
            self.UnitDivided=dictArgs["UnitDivided"] 
            self.UnitFilled=dictArgs["UnitFilled"]
            self.UnitAttributes=dictArgs["UnitAttributes"]  
            self.SectionBlockName=dictArgs["SectionBlockName"]
            self.SectionAttributes=dictArgs["SectionAttributes"]
            self.Material=dictArgs["Material"]      
        except KeyError as e:
            print("mdc.Railing ",self.Handle,"MissingKey",e)            
class MassElement:
    def __init__(self,dictArgs):
        self.StyleName=dictArgs["StyleName"]
        self.Faces=dictArgs["Faces"]
        self.SUWindowGroupID=dictArgs["SUWindowGroupID"]
        self.Type=dictArgs["Type"]
        self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        #--------------------------新增-------------------------------
        self.Material=None #外观材质
        try:
            self.Material=dictArgs["Material"]
        except KeyError as e:
            print("mdc.MassElement ",self.Handle,"MissingKey",e)          
class Window_MassElement(MassElement):
    def __init__(self,dictArgs):
        super().__init__(dictArgs)
        #--------------------------新增-------------------------------
        self.Attributes=None #附加属性:{string:Value}
        self.DWGRef=None #大样图文件:string
        self.SectionBlockName=None #断面图块:string
        self.SectionAttributes=None #断面附加属性:{string:Attribute}
        try:
            self.Attributes=dictArgs["Attributes"]
            self.DWGRef=dictArgs["DWGRef"]
            self.SectionBlockName=dictArgs["SectionBlockName"]
            self.SectionAttributes=dictArgs["SectionAttributes"]        
        except KeyError as e:
            print("mdc.Window_MassElement ",self.Handle,"MissingKey",e)
class Slab: 
    def __init__(self,dictArgs):
        self.StyleName=dictArgs["StyleName"]
        self.Thickness=dictArgs["Thickness"]
        self.Elevation=dictArgs["Elevation"]
        self.Slope=dictArgs["Slope"]
        #self.StartPoint=dictArgs["StartPoint"]
        #self.EndPoint=dictArgs["EndPoint"]
        self.OutLinesPoints=dictArgs["OutLinesPoints"]
        self.LoopsPoints=dictArgs["LoopsPoints"]
        self.Type=dictArgs["Type"] # ="Slab"
        if self.Type==None:self.Type="Slab"
        self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        #--------------------------新增-------------------------------
        self.SlabType=dictArgs["SlabType"]if "SlabType" in dictArgs else dictArgs["slabType"] #==================改名
        # SlabType=屋面"Roof"/楼板"Floor"/楼梯半平台"Platform"/飘窗板"BayWindowTop""BayWindowBottom"/阳台"Balcony"/设备"Equipment"
        self.ID=dictArgs["ID"]
        try:
            self.FloorNo=dictArgs["FloorNo"]
        except KeyError as e:
            print("mdc.Slab ",self.Handle,"MissingKey",e)

class Door_MassElement(MassElement): #门上雨篷
    def __init__(self,dictArgs):
        super().__init__(dictArgs)
        
        self.Attributes=None #附加属性:{string:Value}
        self.DWGRef=None #大样图文件:string
        self.SectionBlockName=None #断面图块:string
        self.SectionAttributes=None #断面附加属性:{string:Attribute}
        try:
            self.Attributes=dictArgs["Attributes"]
            self.DWGRef=dictArgs["DWGRef"]
            self.SectionBlockName=dictArgs["SectionBlockName"]
            self.SectionAttributes=dictArgs["SectionAttributes"] 
        except KeyError as e:
            print("mdc.Door_MassElement ",self.Handle,"MissingKey",e)
class Stair: #楼梯/坡道；以层为单位
    def __init__(self,dictArgs):
        self.StyleName=dictArgs["StyleName"] #楼梯样式:String
        try:
            self.Platform=[] #休息平台:[Slab]
            for platform in dictArgs["vecSlabs"]: #============================改名
                self.Platform.append(Slab(platform))
            self.FlightUp=[] #上行梯段:[Flight] 
            for flight in dictArgs["vecFUFlights"]: #==========================改名
                self.FlightUp.append(Flight(flight))            
            self.FlightDown=[] #下行梯段:[Flight]
            for flight in dictArgs["vecFDFlights"]:#===========================改名
                self.FlightDown.append(Flight(flight)) 
        except:
            self.Platform=[] #休息平台:[Slab]
            for platform in dictArgs["Platform"]: #============================改名
                self.Platform.append(Slab(platform))
            self.FlightUp=[] #上行梯段:[Flight] 
            for flight in dictArgs["FlightUp"]: #==========================改名
                self.FlightUp.append(Flight(flight))            
            self.FlightDown=[] #下行梯段:[Flight]
            for flight in dictArgs["FlightDown"]:#===========================改名
                self.FlightDown.append(Flight(flight)) 
        #self.Handle=dictArgs["Handle"]
        self.SUGroupID=dictArgs["SUGroupID"]
        self.ID=dictArgs["ID"] if "ID" in dictArgs else str(uuid.uuid4())
        self.Type=dictArgs["Type"]
        try:
            self.Name=dictArgs["Name"] #楼梯间名称:String
        except KeyError as e:
            print("mdc.Stair ",self.Handle,"MissingKey",e)        
class Flight: #梯段
    def __init__(self,dictArgs):
        # self.Handle=dictArgs["Handle"]
        self.ID=dictArgs["ID"] if "ID" in dictArgs else str(uuid.uuid4())
        self.Type=dictArgs["Type"]
        self.FlightType=dictArgs["FlightType"] #楼梯="Stair"/坡道="Ramp"/非机动车坡道="BikeRamp"
        try:
            self.StartLine=[dictArgs["StartLine"]["StartPoint"],dictArgs["StartLine"]["EndPoint"]] #起始线:Line3D
            self.EndLine=[dictArgs["EndLine"]["StartPoint"],dictArgs["EndLine"]["EndPoint"]]if dictArgs["EndLine"] is not None else None #终止线:Line3D
            self.FromSlabID=dictArgs["FromSlab"] #起始平台ID
            self.ToSlabID=dictArgs["ToSlab"]  #终止平台ID
        except:
            self.StartLine=dictArgs["StartLine"]
            self.EndLine=dictArgs["EndLine"]
            self.FromSlabID=dictArgs["FromSlabID"] #起始平台ID
            self.ToSlabID=dictArgs["ToSlabID"]  #终止平台ID            
        self.StepWidth=dictArgs["StepWidth"] #踏步宽:Float；坡道无
        self.StepNum=dictArgs["StepNum"] #踏步数:Int；坡道无
        self.Orientation=dictArgs["Orientation"] #起始向量:Vector2D

        self.Height=dictArgs["Height"]
        # self.Components=dictArgs["Components"] #非机动车坡道的组合形式，从高到低从左到右:[{Type:"Ramp"/"Stair",Width:Float}]
        # try:
        # except KeyError as e:
        #     print("mdc.Flight ",self.Handle,"MissingKey",e)
