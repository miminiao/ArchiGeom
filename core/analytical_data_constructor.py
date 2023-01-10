import uuid
from modelling_data_constructor import Stair
import lib.geom as geom
import numpy as np
from typing import TypeVar, List, Tuple
import shapely as sp

T = TypeVar("T")


def getAncestor(obj, expectedType: T, *args, **kwargs) -> T:
    while True:
        if isinstance(obj, expectedType):
            return obj
        if hasattr(obj, "parent") and (obj.parent != None):
            obj = obj.parent
        else:
            return None


def getDescendants(obj, expectedType: T, *args, **kwargs) -> List[T]:
    if isinstance(obj, expectedType):
        return [obj]
    if hasattr(obj, "child") and (len(obj.child) > 0):
        des = []
        for i in obj.child:
            des += getDescendants(i, expectedType)
        return des
    else:
        return []


def getAttributes(obj, expectedType, expectedAttr, *args, **kwargs) -> list:
    if isinstance(obj, expectedType):
        return [getattr(obj, expectedAttr)]
    if hasattr(obj, "child") and (len(obj.child) > 0):
        des = []
        for i in obj.child:
            des += getAttributes(i, expectedType, expectedAttr)
        return des
    else:
        return []


class UniformConfig:
    arg_names = [
        "Land", # 用地配置
        "Parking", # 车位配置
        "Basement", # 地库配置
        "MasterPlan", # 总图配置
        "Building", # 单体配置
        "Material", # 材料配置
        "Profile", # 轮廓形状配置
        "Wall", # 墙配置
        "Door", # 门配置
        "Window", # 窗配置
        "Railing", # 栏杆配置
        "Stair", # 楼梯配置
        "Beam", # 梁配置
        "Slab", # 楼地面配置
        "RoomConfigs",  # 房间默认配置表
        "ConstructionDefinitions",  # 构造做法定义表        
    ]

    def __init__(self, dictArgs, parent=None):
        pass


class Building:
    # 默认从dwgData.json创建
    def __init__(self, dictArgs, dictSlab, parent=None):
        self.vecFrames = []
        if "vecFrames" in dictArgs:
            for floor in dictArgs["vecFrames"]:
                newFloor = Floor(floor, self)
                self.vecFrames.append(newFloor)
        self.AxisGrid = []
        self.Insulation = None
        self.Altitude = None
        self.SectionLine = None  # 剖面线
        self.ID = str(uuid.uuid4())
        if "AxisGrid" not in dictArgs: return
        for axis in dictArgs["AxisGrid"]:  # 轴网
            newAxis = Axis(axis)
            self.AxisGrid.append(newAxis)
        self.InsulationThickness = dictArgs["InsulationThickness"]  # 全局保温厚度
        self.InsulationMaterial = dictArgs["InsulationMaterial"]  # 全局保温厚度
        self.Altitude = dictArgs["Altitude"]  # 绝对高程
        self.SectionLine = dictArgs["SectionLine"]  # 剖面线
        self.guid = dictArgs["guid"]
        self.parent = parent
        self.child = self.vecFrames
        self.ElevationDiff = 0.0  # 室内外高差--------------------------------------修改
        # self.RoofHeight = self.getBuildingHeight(
        #     definition="ROOF"
        # )  # 建筑高度：大屋面--------------------------------------修改
        # self.ParapetHeight = self.getBuildingHeight(
        #     definition="PARAPET"
        # )  # 建筑高度：大屋面女儿墙--------------------------------------修改
        # self.AbsoluteHeight = self.getBuildingHeight(
        #     definition="ABSOLUTE"
        # )  # 建筑高度：最顶部女儿墙--------------------------------------修改

        # try:
        # except KeyError as e:
        #     print("adc.Building ",self.ID,"MissingKey",e)

        self.floor_frame_map=self.get_floor_frame_map()
    # 也可以从已经输出过的OutputModel创建
    @classmethod
    def init_from_output(cls, jsonOutput):
        newBuilding = cls({})
        for floor in jsonOutput["Floors"]:
            newBuilding.vecFrames.append(Floor(floor, newBuilding))
        for axis in jsonOutput["AxisGrid"]:
            newBuilding.AxisGrid.append(Axis(axis))
        newBuilding.Insulation = jsonOutput["Insulation"]
        newBuilding.Altitude = jsonOutput["Altitude"]
        newBuilding.SectionLine = jsonOutput["SectionLine"]
        newBuilding.ID = jsonOutput["ID"]
        return newBuilding
    
    def get_floor_frame_map(self):
        dic={}
        for floor in self.vecFrames:
            floor_list=floor.Frame.get_working_floor_list()
            dic.update({floor:self}for floor in floor_list)
        return dic
    def get_building_height(self, definition="ROOF") -> float:
        """
        返回相应规则定义的建筑高度。

        Parameters
        ----------
        definition : String, optional
            计算规则。The default is "ROOF".

            "ROOF"：室外设计地面至屋面面层。忽略面积<=1/4的凸出部分。
            "PARAPET"：室外设计地面至女儿墙顶。忽略面积<=1/4的凸出部分。
            "ABSOLUTE"：室外设计地面至女儿墙顶。计算至最高部位。

        """
        if definition == "PARAPET":
            buildingHeight = self.ElevationDiff
            for floor in self.vecFrames:
                if floor.Celling.FloorType in ["Normal", "Elevated"]:
                    buildingHeight += len(floor.Frame.getWorkingFloorList()) * float(
                        floor.Frame.WorkingHeight
                    )
                if floor.Celling.FloorType == "Equipment":
                    h = 0.0
                    for roof in floor.Celling.vecRoofs:
                        for wall in roof.Walls:
                            if wall.WallType == "Parapet":
                                h = max(h, wall.Height + wall.Elevation)
                    buildingHeight += h
            return buildingHeight
        elif definition == "ABSOLUTE":
            buildingHeight = self.ElevationDiff
            for floor in self.vecFrames:
                if floor.Celling.FloorType in ["Normal", "Elevated", "Equipment"]:
                    buildingHeight += len(floor.Frame.getWorkingFloorList()) * float(
                        floor.Frame.WorkingHeight
                    )
                if floor.Celling.FloorType == "Roof":
                    h = 0.0
                    for roof in floor.Celling.vecRoofs:
                        for wall in roof.Walls:
                            if wall.WallType == "Parapet":
                                h = max(h, wall.Height + wall.Elevation)
                    buildingHeight += h
            return buildingHeight
        else:  # 含definition=="ROOF"的情况
            buildingHeight = self.ElevationDiff
            for floor in self.vecFrames:
                if floor.Celling.FloorType in ["Normal", "Elevated"]:
                    buildingHeight += len(floor.Frame.getWorkingFloorList()) * float(
                        floor.Frame.WorkingHeight
                    )
            return buildingHeight


class Floor:
    def __init__(self, dictArgs, dictSlab, parent=None):
        self.Celling = Celling(dictArgs["Celling"],dictSlab, self)
        self.Frame = Frame(dictArgs["Frame"])
        self.parent = parent
        self.child = self.Celling.Structs

class Frame:
    def __init__(self, dictArgs):
        # self.FrameLayer=dictArgs["FrameLayer"]
        # self.SideDistance=dictArgs["SideDistance"]
        self.WorkingName = dictArgs["WorkingName"]
        self.WorkingNum = dictArgs["WorkingNum"]
        self.WorkingType = dictArgs["WorkingType"]
        self.WorkingHeight = dictArgs["WorkingHeight"]
        self.WorkingFloor = dictArgs["WorkingFloor"]
        self.Odevity = dictArgs["Odevity"]
        # self.BlockId=dictArgs["BlockId"]
        # self.extents=dictArgs["extents"]
        self.Xpoint = dictArgs["Xpoint"]
        # try:
        # except KeyError as e:
        #     print("adc.Frame",self.WorkingName,"MissingKey",e)
        self.working_floor_list=self.get_working_floor_list()

    def get_working_floor_list(self):
        res=[]
        segs = self.WorkingFloor.replace(" ","").split(",")
        for seg in segs:
            if "-" in seg:
                bounds=seg.split("-")
                res.extend(range(int(bounds[0]),int(bounds[1])+1))
            else: res.append(int(seg))
        res.sort()
        return res
    def get_neighbor_frames(self):
        building=getAncestor(self,Building)
        upper=self.working_floor_list[-1]+1
        lower=self.working_floor_list[0]-1
        neighbor_frames=[]
        for num in (upper,lower):
            if num in building.floor_frame_map:
                neighbor_frames.append(building.floor_frame_map[num])
            else: neighbor_frames.append(None)
        return neighbor_frames

class Celling:
    def __init__(self, dictArgs,dictSlab, parent=None):
        self.FloorCode = dictArgs["FloorCode"]
        self.FloorNum = dictArgs["FloorNum"]
        self.Height = dictArgs["Height"]
        self.Area = dictArgs["Area"]
        self.Name = dictArgs["Name"]
        # self.BasePoint=dictArgs["BasePoint"]
        self.FloorRate = dictArgs["FloorRate"]
        self.Structs = []
        for struct in dictArgs["Structs"]:
            newStruct = Struct(struct, self)
            self.Structs.append(newStruct)
        self.BuildingArea = dictArgs["BuildingArea"]
        self.SumBalconyArea = dictArgs["SumBalconyArea"]
        self.SumInsideArea = dictArgs["SumInsideArea"]
        # self.vecFloors=dictArgs["vecFloors"]
        # self.vecCores=dictArgs["vecCores"]
        self.ID = dictArgs["ID"]
        self.vecOutWalls = dictArgs["vecOutWalls"]
        # self.vecOutWall=dictArgs["vecOutWall"]
        # self.SingleMaterialWalls=dictArgs["SingleMaterialWalls"]
        # self.DoubleMaterialWalls=dictArgs["DoubleMaterialWalls"]
        # self.BlueWalls=dictArgs["BlueWalls"]
        self.vecRoofs = []
        for roof in dictArgs["vecRoofs"]:
            newRoof = Room(roof, self)
            self.vecRoofs.append(newRoof)
        self.DeformationJoint = dictArgs["DeformationJoint"]  # 变形缝区域
        self.Outline = dictArgs["Outline"]  # 楼层外轮廓
        self.FloorType = dictArgs["FloorType"]  # 普通层="Normal",机房层="Equipment",屋顶层="Roof"
        self.vecSlabs=[]
        self.parent = parent
        self.child = self.Structs
        # try:
        # except KeyError as e:
        #     print("adc.Celling",self.ID,"MissingKey",e)

def gen_slabs(building:Building, dictACASlab:dict):
    for floor in building.vecFrames:
        celling=floor.Celling
        roof_areas=[]
        # 生成楼面、阳台、设备平台、飘窗板      
        rooms=getDescendants(celling,Room)  
        room_slab_type_dict={
            "阳台":("Balcony",-20,100),
            "设备平台":("Equipment",-100,30),
            "电井":("Floor",0,30),
            "电梯厅":("Floor",0,30),
            "合用前室":("Floor",0,30),
            "水暖井":("Floor",0,30),
            "else":("Floor",0,80),
            "厨房":("Floor",0,100),
            "卫生间":("Floor",-20,100),
            "飘窗":(("BayWindowTop",-100,30),("BayWindowBottom",100,30),None),
            "屋面":("Roof",200,200),
            "露台":("Roof",-50,150),
        }
        for room in rooms:
            if room.roomName in room_slab_type_dict:
                room_config=room_slab_type_dict[room.roomName] 
            else: room_slab_type_dict["else"]
            slab_args={"ID":uuid.uuid4(),
                 "vecOutlinepts":room.profile["Outline"],
                 "vecHolespts":room.profile["Holes"],
                 "_thickness":room_config[2],
                 "_styleName":None,
                 "_slabType":room_config[0],
                 "_elevation":room_config[1],
                 "_roomType":room.roomName,
                 "_offsetZ":0.0,
                 "_inclinagtion":0.0,
                 }
            if room.roomName=="飘窗":
                # 生成飘窗上下板
                elev=[1e6,-1e6]
                for wall in room.Walls:
                    elev[0]=min(elev[0],wall.Elevation+wall.Height+room_config[0][1])
                    elev[1]=max(elev[1],wall.Elevation+room_config[1][1])
                for i in range(2):
                    slab_args["ID"]=uuid.uuid4()
                    slab_args["_slabType"]=room_config[i][0]
                    slab_args["_elevation"]=elev[i]
                    slab_args["_thickness"]=room_config[i][2]
                    new_slab=ACASlab(slab_args,parent=celling)
                    celling.vecSlab.append(new_slab)
            elif room.roomName=="屋面":
                # 单独拿出屋面，后续一并处理
                poly=sp.Polygon(
                    [(pt["X"],pt["Y"]) for pt in room.profile["Outline"]],
                    [[(pt["X"],pt["Y"])for pt in hole] for hole in room.profile["Holes"]]
                ).simplify()
                roof_areas.append(poly)
            else:
                new_slab=ACASlab(slab_args,parent=celling)
                celling.vecSlabs.append(new_slab)
        # 生成屋面板
        for frame in dictACASlab:
            if frame["FrameName"]==celling.Name: break
        ## 拿出ACASlab.json里记录的屋面板
        for slab in frame["vecSlabs"]:
            if slab["_slabType"]=="Roof":
                poly=sp.Polygon(
                    [(pt["X"],pt["Y"]) for pt in slab["vecOutlinepts"]],
                    [[(pt["X"],pt["Y"])for pt in hole] for hole in slab["vecHolespts"]]
                ).simplify()
                roof_areas.append(poly)
        ## 计算上下层的差polygon
        roof_poly=sp.Polygon(
            [(pt["X"],pt["Y"]) for pt in frame["vecAreaLines"]],
            [[(pt["X"],pt["Y"])for pt in hole] for hole in frame["vecOllowLines"]]
        )
        lower_frame=floor.Frame.get_neighbor_frames[1]
        if lower_frame is not None:
            for frame in dictACASlab:
                if frame["FrameName"]==lower_frame.WorkingName:
                    break
            lower_area_poly=sp.Polygon(
                [(pt["X"],pt["Y"]) for pt in frame["vecAreaLines"]],
                [[(pt["X"],pt["Y"])for pt in hole] for hole in frame["vecOllowLines"]]
            )
            roof_poly=roof_poly.difference(lower_area_poly)
        if isinstance(roof_poly,sp.MultiPolygon):
            for poly in roof_poly.geoms:
                if poly.area>1e-3:
                    roof_areas.append(poly)
        elif roof_poly.area>1e-3:
            roof_areas.append(roof_poly)
        ## 创建所有的屋面板对象
        room_config=room_slab_type_dict["屋面"] 
        for poly in roof_areas:
            slab_args={"ID":uuid.uuid4(),
                 "vecOutlinepts":[{"X":pt[0],"Y":pt[1]} for pt in poly.exterior.coords],
                 "vecHolespts":[[{"X":pt[0],"Y":pt[1]} for pt in ring] for ring in poly.interiors],
                 "_thickness":room_config[2],
                 "_styleName":None,
                 "_slabType":room_config[0],
                 "_elevation":room_config[1],
                 "_roomType":room.roomName,
                 "_offsetZ":0.0,
                 "_inclinagtion":0.0,
                 }
class ACASlab:
    def __init__(self,dictArgs,parent=None) -> None:
        self.ID=dictArgs["ID"]
        self.vecOutlinepts=dictArgs["vecOutlinepts"]
        self.vecHolespts=dictArgs["vecHolespts"]
        self.thickness=dictArgs["thickness"]
        self.styleName=dictArgs["styleName"]
        self.slabType=dictArgs["slabType"]
        self.elevation=dictArgs["elevation"]
        self.roomType=dictArgs["roomType"]
        self.offsetZ=dictArgs["offsetZ"]
        self.inclination=dictArgs["incliation"]
        self.arch_fin_thickness=dictArgs["arch_fin_thickness"]
        self.parent=parent

class Struct:
    def __init__(self, dictArgs, parent=None):
        self.StructID = dictArgs["StructID"]
        self.Units = []
        for unit in dictArgs["Units"]:
            newUnit = Unit(unit, self)
            self.Units.append(newUnit)
        self.parent = parent
        self.child = self.Units


class Unit:
    def __init__(self, dictArgs, parent=None):
        self.UnitCode = dictArgs["UnitCode"]
        self.FloorCompound = dictArgs["FloorCompound"]
        self.Familys = []
        for family in dictArgs["Familys"]:
            newFamily = Family(family, self)
            self.Familys.append(newFamily)
        self.CoreTubes = []
        for coreTube in dictArgs["CoreTubes"]:
            newCoreTube = CoreTube(coreTube, self)
            self.CoreTubes.append(newCoreTube)
        self.ID = dictArgs["ID"]
        self.parent = parent

    @property
    def child(self):
        return self.Familys + self.CoreTubes


class Family:
    def __init__(self, dictArgs, parent=None):
        self.Code = dictArgs["Code"]
        self.FamilyName = dictArgs["FamilyName"]
        self.Setting = dictArgs["Setting"]
        self.Position = dictArgs["Position"]
        self.Width = dictArgs["Width"]
        self.Depth = dictArgs["Depth"]
        self.Area = dictArgs["Area"]
        self.BayNum = dictArgs["BayNum"]
        self.IsPenetrating = dictArgs["IsPenetrating"]
        self.Feature = dictArgs["Feature"]
        self.ACType = dictArgs["ACType"]
        self.Rooms = []
        for room in dictArgs["Rooms"]:
            newRoom = Room(room, self)
            self.Rooms.append(newRoom)
        self.ModelArea = dictArgs["ModelArea"]
        self.ModelBuildingArea = dictArgs["ModelBuildingArea"]
        self.BalconyArea = dictArgs["BalconyArea"]
        self.No = dictArgs["No"]
        self.TransverseHall = dictArgs["TransverseHall"]
        self.SuiteNum = dictArgs["SuiteNum"]
        self.KitchenConfiguration = dictArgs["KitchenConfiguration"]
        self.MainBathroomConfiguration = dictArgs["MainBathroomConfiguration"]
        self.XuanguanArea = dictArgs["XuanguanArea"]
        self.PublicArea = dictArgs["PublicArea"]
        self.ID = dictArgs["ID"]
        self.SumBalconyHalf = dictArgs["SumBalconyHalf"]
        self.SumBalconyFull = dictArgs["SumBalconyFull"]
        self.vecInDoors = dictArgs["vecInDoorsHandle"]
        self.FloorCompound = dictArgs["FloorCompound"]
        # self.Outline=dictArgs["Outline"]
        # self.vecAllHandles=dictArgs["vecAllHandles"]
        self.parent = parent
        self.child = self.Rooms


class CoreTube:
    def __init__(self, dictArgs, parent=None):
        self.NameId = dictArgs["NameId"]
        self.Code = dictArgs["Code"]
        self.CoreTubeName = dictArgs["CoreTubeName"]
        self.ElevatorNum = dictArgs["ElevatorNum"]
        self.ElevatorNo = dictArgs["ElevatorNo"]
        self.LadderWidth = dictArgs["LadderWidth"]
        self.TotalArea = dictArgs["TotalArea"]
        self.StairNum = dictArgs["StairNum"]
        self.Feature = dictArgs["Feature"]
        self.Rooms = []
        for room in dictArgs["Rooms"]:
            newRoom = Room(room, self)
            self.Rooms.append(newRoom)
        # self.vecFloors=dictArgs["vecFloors"]
        # self.Outline=dictArgs["Outline"]
        self.vecInDoors = dictArgs["vecInDoors"]
        self.ID = dictArgs["ID"]
        self.parent = parent
        self.child = self.Rooms


class Room:
    def __init__(self, dictArgs, parent=None):
        self.NameId = dictArgs["NameId"]
        self.vecGraphs = dictArgs["vecGraphs"]
        self.vecFurnitures = []
        for furniture in dictArgs["vecFurnitures"]:
            newFurniture = Furniture(furniture, self)
            self.vecFurnitures.append(newFurniture)
        self.Id = dictArgs["Id"]
        self.AreaRight = dictArgs["AreaRight"]
        self.RoomName = dictArgs["RoomName"]
        ###############
        if self.RoomName=="Multiple":
            self.RoomType=self.vecAllNames[0]
        elif self.RoomName=="":
            self.RoomType="自定义"
        else:
            self.RoomType=self.RoomName
        self.RoomConfig=None
        self.Regions=[]
        ###############
        self.Furnitures = dictArgs["Furnitures"]
        self.NetWidth = dictArgs["NetWidth"]
        self.NetDepth = dictArgs["NetDepth"]
        self.RoomArea = dictArgs["RoomArea"]
        self.Orientation = dictArgs["Orientation"]
        self.Walls = []
        for wall in dictArgs["Walls"]:
            newWall = Wall(wall, self)
            self.Walls.append(newWall)
        self.vecHandRails = []
        for handrail in dictArgs["vecHandRails"]:
            newHandrail = Handrail(handrail, self)
            self.vecHandRails.append(newHandrail)
        self.vecAllName = dictArgs["vecAllName"]
        self.vecAllTextIds = dictArgs["vecAllTextIds"]
        self.Width = dictArgs["Width"]
        self.Depth = dictArgs["Depth"]
        self.BayWindowDepth = dictArgs["BayWindowDepth"]
        self.OpeningWidth = dictArgs["OpeningWidth"]
        self.WidthNum = dictArgs["WidthNum"]
        self.ObjectID = dictArgs["ObjectID"]
        self.ObjectAllTextID = dictArgs["ObjectAllTextID"]
        self.vecAllHandles = dictArgs["vecAllHandles"]
        self.Profile = {
            "Outline": dictArgs["Profile"],
            "Holes": dictArgs["Holes"],
        } 
        self.Stair = (
            Stair(dictArgs["Stair"]) if dictArgs["Stair"] is not None else None
        )  # 所包含的楼梯对象
        self.vecAllTextCoords = dictArgs["vecAllTextCoords"]  # 所有文字的坐标OK
        self.parent = parent
        self.neighbor = []  # 通过门相连的房间
        # try:
        # except KeyError as e:
        #     print("adc.Room ",self.ObjectID,"MissingKey",e)

    @property
    def child(self):
        return self.Walls + self.vecHandRails + self.vecFurnitures

    def getNextRoom(self, commonDoor: "Door") -> "Room":
        doors = getDescendants(getAncestor(self, Celling), Door)
        for door in doors:
            if (door.Handle == commonDoor.Handle) and (door.parent.parent is not self):
                return door.parent.parent

    def getPriority(self) -> int:
        allName = ",".join(self.vecAllName)
        if self.AreaRight == 0:
            return 1
        elif "廊" in allName:
            return 2
        elif "楼梯" in allName:
            return 3
        elif "前室" in allName:
            return 4
        else:
            return 5


class Wall:
    def __init__(self, dictArgs, parent: Room = None):
        self.Id = dictArgs["Id"]
        self.ObjectID = dictArgs["ObjectID"]
        self.StyleName = dictArgs["StyleName"]
        self.Width = dictArgs["Width"]
        self.Height = dictArgs["Height"]
        self.Length = dictArgs["Length"]
        self.Alignment = dictArgs["Alignment"]
        self.Revolve = dictArgs["Revolve"]
        self.Elevation = dictArgs["Elevation"]
        self.StartPoint = dictArgs["StartPoint"]
        self.EndPoint = dictArgs["EndPoint"]
        self.Sidetags = dictArgs["Sidetags"]
        self.Orientation = dictArgs["Orientation"]
        self.Doors = []
        for door in dictArgs["Doors"]:
            newDoor = Door(door, self)
            self.Doors.append(newDoor)
        self.Windows = []
        for window in dictArgs["Windows"]:
            newWindow = Window(window, self)
            self.Windows.append(newWindow)
        self.WWGroups = []
        for wwGroup in dictArgs["WWGroups"]:
            newWWGroup = WWGroup(wwGroup, self)
            self.WWGroups.append(newWWGroup)
        self.Handle = dictArgs["Handle"]
        self.AdjacentRoom = dictArgs["AdjacentRoom"]
        self.WallType = dictArgs[
            "WallType"
        ]  # 墙类型：普通墙="Normal"/矮墙="Low"/女儿墙="Parapet"/挡土墙="Retaining"
        self.parent = parent
        # try:
        # except KeyError as e:
        #     print("mc.Wall ",self.ObjectID,"MissingKey",e)

    @property
    def child(self):
        return self.Doors + self.Windows + self.WWGroups


class Window:
    def __init__(self, dictArgs, parent: Wall = None):
        self.StyleName = dictArgs["StyleName"]
        self.Type = dictArgs["Type"]
        self.Width = dictArgs["Width"]
        self.Width2 = dictArgs["Width2"]
        self.Height = dictArgs["Height"]
        self.SillHeight = dictArgs["SillHeight"]
        self.Xaxis = dictArgs["Xaxis"]
        self.Yaxis = dictArgs["Yaxis"]
        self.Zaxis = dictArgs["Zaxis"]
        self.AxisReverse = dictArgs["AxisReverse"]
        self.Id = dictArgs["Id"]
        self.AnchoredCurveId = dictArgs["AnchoredCurveId"]
        self.ObjectID = dictArgs["ObjectID"]
        self.Handle = dictArgs["Handle"]
        self.parent = parent


class Door:
    def __init__(self, dictArgs, parent: Wall = None):
        self.StyleName = dictArgs["StyleName"]
        self.Width = dictArgs["Width"]
        self.Height = dictArgs["Height"]
        self.Xaxis = dictArgs["Xaxis"]
        self.Yaxis = dictArgs["Yaxis"]
        self.Zaxis = dictArgs["Zaxis"]
        self.AxisReverse = dictArgs["AxisReverse"]
        self.Id = dictArgs["Id"]
        self.AnchoredCurveId = dictArgs["AnchoredCurveId"]
        self.ObjectID = dictArgs["ObjectID"]
        self.Handle = dictArgs["Handle"]
        self.parent = parent

    def getAdjacentRooms(self) -> Tuple[Room, Room]:
        """
        获取当前门两侧的房间

        Returns
        -------
        (Room,Room)
            按开启方向返回(fromRoom,toRoom)

        """
        wall = self.parent
        startPoint = np.array([wall.StartPoint["X"], wall.StartPoint["Y"]])
        endPoint = np.array([wall.EndPoint["X"], wall.EndPoint["Y"]])
        thisRoom = wall.parent
        otherRoom = thisRoom.getNextRoom(self)
        vectorWall = endPoint - startPoint
        nearestPointOnThisRoom = geom.getNearestPointFromPointToPolygon(
            (self.Xaxis, self.Yaxis),
            thisRoom.Profile["Outline"],
            thisRoom.Profile["Holes"],
        )
        vectorThisRoom = np.array(nearestPointOnThisRoom) - startPoint
        if (np.cross(vectorWall, vectorThisRoom) > 0) ^ (self.AxisReverse["Y"]):
            return otherRoom, thisRoom
        else:
            return thisRoom, otherRoom


class WWGroup:
    def __init__(self, dictArgs, parent=None):
        self.StyleName = dictArgs["StyleName"]
        self.Width = dictArgs["Width"]
        self.Height = dictArgs["Height"]
        self.Xaxis = dictArgs["Xaxis"]
        self.Yaxis = dictArgs["Yaxis"]
        self.Zaxis = dictArgs["Zaxis"]
        self.AxisReverse = dictArgs["AxisReverse"]
        self.parent = parent


class Handrail:
    def __init__(self, dictArgs, parent=None):
        self.StyleName = dictArgs["StyleName"]
        self.Height = dictArgs["Height"]
        self.Elevation = dictArgs["Elevation"]
        self.BaseLine = dictArgs["BaseLine"]
        self.parent = parent
        self.Handle = dictArgs["Handle"]
        # try:
        # except KeyError as e:
        #     print("adc.Handrail ","MissingKey",e)


class Furniture:
    def __init__(self, dictArgs, parent=None):
        self.Id = dictArgs["Id"]
        self.ObjectID = dictArgs["ObjectID"]
        self.Handle = dictArgs["Handle"]
        # self.Name=dictArgs["Name"]
        # --------------------------------新增--------------------------------
        self.EffectiveName = dictArgs["EffectiveName"]  # 块定义名
        self.InsertionPoint = dictArgs["InsertionPoint"]  # 图块插入点OK
        self.Rotation = dictArgs["Rotation"]  # 图块转角OK
        self.ScaleX = dictArgs["ScaleX"]  # 图块缩放比例OK
        self.ScaleY = dictArgs["ScaleY"]
        self.ScaleZ = dictArgs["ScaleZ"]
        self.Attributes = dictArgs["Attributes"]  # 块属性OK
        self.Parameters = dictArgs["Parameters"]  # 块动态参数
        self.Type = dictArgs["Type"]  # 家具类型，http://172.16.0.230:8090/x/84Fg
        self.parent = parent

        # try:
        # except KeyError as e:
        #     print("adc.Furniture ",self.ObjectID,"MissingKey",e)

    @classmethod
    def getTypeByEffectiveName(cls, strName: str) -> str:
        typeList = strName.split("_")
        if typeList[1] == "EQPM":
            if (typeList[2] == "KICH") or (typeList[2] == "BATH"):
                entType = typeList[3]
            else:
                entType = typeList[2]
        elif typeList[1] == "FURN":
            entType = typeList[2]
        else:
            entType = "ROOM"
        return entType


class Axis:  # 轴线
    def __init__(self, dictArgs):
        self.startPoint = None  # 轴线起点
        self.endPoint = None  # 轴线终点
        self.number = None  # 轴线编号
        # self.ID=str(uuid.uuid4())
        try:
            self.startPoint = dictArgs["startPoint"]
            self.endPoint = dictArgs["endPoint"]
            self.number = dictArgs["number"]
            # self.ID=dictArgs["ID"]
        except KeyError as e:
            print("adc.Axis ", self.Number, "MissingKey", e)


class BayWindow:
    pass


class FurnitureStyle:  # 家具图块定义
    def __init__(self, dictArgs):
        self.Name = dictArgs["Name"]  # 图块名
        self.Type = dictArgs["Type"]  # 家具类型
        self.Profile = dictArgs["Profile"]  # 边界轮廓
        self.EmbeddedBlocks = dictArgs["EmbeddedBlocks"]  # 预埋点位
        # try:
        # except KeyError as e:
        #     print("adc.FurnStyle ","MissingKey",e)


class Test:
    def __init__(self) -> None:
        self.WorkingFloor="1,2,3-7,11-13,5,8-9"
    def getWorkingFloorList(self):
        res=[]
        segs = self.WorkingFloor.replace(" ","").split(",")
        for seg in segs:
            if "-" in seg:
                bounds=seg.split("-")
                res.extend(range(int(bounds[0]),int(bounds[1])+1))
            else: res.append(int(seg))
        res.sort()
        return res

if __name__=="__main__":
    t=Test()
    print(t.getWorkingFloorList())