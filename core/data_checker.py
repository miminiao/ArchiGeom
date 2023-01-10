import modelling_data_constructor as mdc
import analytical_data_constructor as adc
import matplotlib.pyplot as plt
import numpy as np
import lib.geom as geom
from typing import List
from data_binder import dataBuilding,modelBuilding

#basePoint=[[71500.00,-26300.00],[71500.00,13400.00],[71500.00,53100.00],[71500.00,92800.00],[0,0]]
MAX_NUM=1e9
#%% 数据完整性检查
#%% 检查外墙
def drawOutWalls(frameName="标准层",label="vecOutWalls"):
    for floor in dataBuilding.vecFrames:
        if frameName not in floor.Celling.Name:continue
        fig = plt.figure(figsize=(30, 30))
        ax = plt.gca()
        ax.set_aspect(1)
        for wall in adc.getDescendants(dataBuilding, adc.Wall):
            if wall.Handle not in getattr(floor.Celling, label): continue
            x,y=[wall.StartPoint["X"],wall.EndPoint["X"]],[wall.StartPoint["Y"],wall.EndPoint["Y"]]
            plt.plot(x+x[:1],y+y[:1],color=np.random.rand(3,),linewidth=1)                            

#%% 检查房间
def drawRooms(frameName="标准层"):
    for floor in dataBuilding.vecFrames:
        if frameName not in floor.Celling.Name:continue
        fig = plt.figure(figsize=(30, 30))
        ax = plt.gca()
        ax.set_aspect(1)
        for room in adc.getDescendants(dataBuilding, adc.Room):
            if adc.getAncestor(room, adc.Floor) != floor: continue
            x,y=[],[]
            for point in room.Profile["Outline"]:
                x.append(point["X"])
                y.append(point["Y"])
            plt.plot(x+x[:1],y+y[:1],color=np.random.rand(3,),linewidth=1)                            
            for hole in room.Profile["Holes"]:
                x,y=[],[] 
                for point in hole:
                    x.append(point["X"])
                    y.append(point["Y"])
                plt.plot(x+x[:1],y+y[:1],color=np.random.rand(3,),linewidth=1)

#%% 检查轮廓
def drawOutline(frameName="标准层"):
    for floor in dataBuilding.vecFrames:
        if frameName not in floor.Celling.Name:continue
        fig = plt.figure(figsize=(30, 30))
        ax = plt.gca()
        ax.set_aspect(1)
        for poly in floor.Celling.DeformationJoint:
            x,y=[],[]
            for point in poly:
                x.append(point["X"])
                y.append(point["Y"])
            plt.plot(x+x[:1],y+y[:1],color=np.random.rand(3,),linewidth=1)  
        for poly in floor.Celling.Outline:
            x,y=[],[]
            for point in poly:
                x.append(point["X"])
                y.append(point["Y"])
            plt.plot(x+x[:1],y+y[:1],color=np.random.rand(3,),linewidth=1)
  
#%%检查楼梯
def drawStairs(building=dataBuilding, styleName="首层"):
    def drawStair(stair:mdc.Stair):
        def drawFlight(flight:mdc.Flight):
            def drawLine(line):
                plt.plot([line["StartPoint"]["X"],line["EndPoint"]["X"]],
                         [line["StartPoint"]["Y"],line["EndPoint"]["Y"]],
                         linewidth=3)
                mid=np.array([(line["StartPoint"]["X"]+line["EndPoint"]["X"])/2,
                              (line["StartPoint"]["Y"]+line["EndPoint"]["Y"])/2,
                              (line["StartPoint"]["Z"]+line["EndPoint"]["Z"])/2])
                plt.scatter(mid[0],mid[1])
                plt.text(mid[0],mid[1],mid[2],fontsize=7)
                plt.plot([mid[0],mid[0]+flight.Orientation["X"]*500],
                         [mid[1],mid[1]+flight.Orientation["Y"]*500])
            if (flight.StartLine is not None) and (len(flight.StartLine)>0):
                drawLine(flight.StartLine)
            if (flight.EndLine is not None) and (len(flight.EndLine))>0:
                drawLine(flight.EndLine)        
        if styleName not in stair.styleName: return
        for slab in stair.Platform:
            x,y=[],[]
            for point in slab.OutLinesPoints:
                x.append(point["X"])
                y.append(point["Y"])
            plt.plot(x+x[:1],y+y[:1],color=np.random.rand(3,),linewidth=1)
            plt.text(np.array(x).mean(),np.array(y).mean(),int(slab.Elevation),fontsize=7)
        for flight in stair.FlightUp:
            drawFlight(flight)
        for flight in stair.FlightDown:
            drawFlight(flight)        

    fig = plt.figure(figsize=(30, 30))
    ax = plt.gca()
    ax.set_aspect(1)
    if isinstance(building,adc.Building):
        rooms=adc.getDescendants(dataBuilding, adc.Room)
        for room in rooms:
            if room.Stair is not None: 
                drawStair(room.Stair)  
    else:
        for stair in modelBuilding.getStairs():
            drawStair(stair)



#%% 检查楼板
def drawSlabs(slabType="Balcony",floorNo=1):
    fig = plt.figure(figsize=(30, 30))
    ax = plt.gca()
    ax.set_aspect(1)
    count=0
    for slab in modelBuilding.Slabs:
        if (slabType!="all")and(slabType not in slab.SlabType):continue
        if slab.FloorNo!=floorNo:continue
        count+=1
        x,y=[],[]
        for point in slab.OutLinesPoints:
            x.append(point["X"])
            y.append(point["Y"])
        plt.fill(x,y,color="b",alpha=0.3)
        for hole in slab.LoopsPoints:
            x,y=[],[] 
            for point in hole:
                x.append(point["X"])
                y.append(point["Y"])
            plt.fill(x,y,color="w",alpha=0.7)
    return count


#%% 业务逻辑检查
#%% 检查器
checkers=[]
class Checker():
    def __init__(self,dataClass,typeToBeChecked,filterFunc):
        self.typeToBeChecked=typeToBeChecked
        self.getDescendantsFunc=dataClass.getDescendantsFunc
        self.filterFunc=filterFunc
        self.failedObj=[]
    def check(self,root):
        self.failedObj=[]
        for item in self.getDescendantsFunc(root, self.typeToBeChecked):
            if not self.filterFunc(item):
                self.failedObj.append(item)
        return self.failedObj
#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: GB 50096-2011
    NAME: 住宅设计规范
    ITEM: 5.2.1
    CONTENT: 
    卧室的使用面积应符合下列规定：
    1 双人卧室不应小于9m2；
    2 单人卧室不应小于5m2；
    3 兼起居的卧室不应小于12m2。
    """

    if "卧" in ",".join(item.vecAllName):
        if "兼起居" in ",".join(item.vecAllName):
            if item.RoomArea<12.0:
                return False
            else: 
                return True
        if "THSTD_FURN_BED2" in ",".join(item.Furnitures) and item.RoomArea<9.0:
            return False
        if item.RoomArea<5.0:
            return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: GB 50096-2011
    NAME: 住宅设计规范
    ITEM: 5.2.2
    CONTENT: 
    起居室（厅）的使用面积不应小于10m2。
    """

    if "客厅" in ",".join(item.vecAllName) and item.RoomArea<10.0:
        return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: GB 50096-2011
    NAME: 住宅设计规范
    ITEM: 5.3.1
    CONTENT: 
    厨房的使用面积应符合下列规定：
    1 由卧室、起居室（厅）、厨房和卫生间等组成的住宅套型的厨房使用面积，不应小于4.0m2；
    2 由兼起居的卧室、厨房和卫生间等组成的住宅最小套型的厨房使用面积，不应小于3.5m2。
    """

    if "厨" in ",".join(item.vecAllName):
        for room in adc.getDescendants(item.parent,adc.Room):
            if "兼起居" in ",".join(room.vecAllName):
                if item.RoomArea<3.5:
                    return False
                else:
                    return True
        if item.RoomArea<4.0:
            return False
    return True
checkers.append(Checker(adc.Family,filterFunc))

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: 
    NAME: 住宅项目规范
    ITEM: 4.1.2
    CONTENT: 
    厨房的使用面积不应小于3.5m2。
    """

    if "厨" in ",".join(item.vecAllName) and item.RoomArea<3.5:
        return False
    return True
checkers.append(Checker(adc.Family,filterFunc))

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: GB 50096-2011
    NAME: 住宅设计规范
    ITEM: 5.4.1
    CONTENT: 
    三件卫生设备集中配置的卫生间的使用面积不应小于2.5m2。
    """

    if "卫" in ",".join(item.vecAllName):
        if "THSTD_EQPM_BATH_TOLT" in ",".join(item.Furnitures)\
            and "THSTD_EQPM_BATH_SINK" in ",".join(item.Furnitures)\
            and ("THSTD_EQPM_BATH_SHWR" in ",".join(item.Furnitures)\
                or "THSTD_EQPM_BATH_TUB" in ",".join(item.Furnitures)):
            if item.RoomArea<2.5:
                return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: GB 50096-2011
    NAME: 住宅设计规范
    ITEM: 5.4.2
    CONTENT: 
    卫生间可根据使用功能要求组合不同的设备。不同组合的空间使用面积应符合下列规定：
    1 设便器、洗面器时不应小于1.80m2；
    2 设便器、洗浴器时不应小于2.00m2；
    3 设洗面器、洗浴器时不应小于2.00m2；
    4 设洗面器、洗衣机时不应小于1.80m2；
    5 单设便器时不应小于1.10m2。
    """

    if "卫" in ",".join(item.vecAllName):
        furTypes=adc.getAttributes(item,adc.Furniture,"Type")
        n=("WASH" in furTypes)*16+("TUB" in furTypes)*8+("SHWR" in furTypes)*4+("SINK" in furTypes)*2+("TOLT" in furTypes)
        if (n==3)and(item.RoomArea<1.8):return False
        if (n==5 or n==9)and(item.RoomArea<2.0):return False
        if (n==6 or n==10)and(item.RoomArea<2.0):return False
        if (n==18)and(item.RoomArea<1.8):return False
        if (n==1)and(item.RoomArea<1.1):return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% OK
def filterFunc(item:adc.Family):
    """
    CODE: GB 50096-2011
    NAME: 住宅设计规范
    ITEM: 5.1.2
    CONTENT: 
    套型的使用面积应符合下列规定：
    1 由卧室、起居室（厅）、厨房和卫生间等组成的套型，其使用面积不应小于30m2；
    2 由兼起居的卧室、厨房和卫生间等组成的最小套型，其使用面积不应小于22m2。
    """

    for room in item.Rooms:
        if "卧" in ",".join(room.vecAllName) and "兼起居" in ",".join(room.vecAllName):
            if item.Area<22:
                return False
            else:
                return True
    if (item.Area>0)and(item.Area<30):
        return False
    return True
checkers.append(Checker(adc.Family,filterFunc))

#%% OK
def filterFunc(item:adc.Family):
    """
    CODE: GB 50763-2012
    NAME: 无障碍设计规范
    ITEM: 3.12.4
    CONTENT: 
    无障碍住房及宿舍的其他规定：
    1 单人卧室面积不应小于 7.00 m2 ，
    双人卧室面积不应小于 10.50 m2，
    兼起居室的卧室面积不应小于 16.00 m2，
    起居室面积不应小于 14.00 m2，
    厨房面积不应小于 6.00 m2；
    2 设坐便器、洗浴器（浴盆或淋浴）、洗面盆三件卫生洁具的卫生间面积不应小于 4.00 m2；
    设坐便器、洗浴器二件卫生洁具的卫生间面积不应小于 3.00 m2；
    设坐便器、洗面盆二件卫生洁具的卫生间面积不应小于 2.50 m2；
    单设坐便器的卫生间面积不应小于 2.00 m2；
    """

    for r in item.Rooms:
        if ("无障碍" in ",".join(r.vecAllName)):
            for room in item.Rooms:
                furTypes=adc.getAttributes(room,adc.Furniture,"Type")
                if ("厨房" in ",".join(room.vecAllName))and(room.RoomArea<6.0):return False
                if ("卧室" in ",".join(room.vecAllName)):
                    n=("BED2" in furTypes)*2+("BED1" in furTypes)
                    if (n==1)and(room.RoomArea<7.0):return False
                    if (n==2)and(room.RoomArea<10.5):return False
                if ("客厅" in ",".join(room.vecAllName))and(room.RoomArea<14.0):return False
                if ("卧室兼起居" in ",".join(room.vecAllName))and(room.RoomArea<16.0):return False
                if ("卫" in ",".join(room.vecAllName)):
                    n=("TUB" in furTypes)<<3+("SHWR" in furTypes)<<2+("SINK" in furTypes)<<1+("TOLT" in furTypes)
                    if (n>3)and(room.RoomArea<4.0):return False
                    if (n==5 or n==9)and(room.RoomArea<3.0):return False
                    if (n==3)and(room.RoomArea<2.5):return False
                    if (n==1)and(room.RoomArea<2.0):return False


    return True
checkers.append(Checker(adc.Family,filterFunc))            

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: 
    NAME: 住宅项目规范
    ITEM: 4.1.1
    CONTENT: 
    卧室的使用面积应符合下列规定：
    1 卧室使用面积不应小于6m2；
    2 兼起居的卧室使用面积不应小于9m2。
    """

    if "卧" in ",".join(item.vecAllName):
        if item.RoomArea<6.0:
            return False
        if "兼起居" in ",".join(item.vecAllName) and item.RoomArea<12.0:
            return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% OK
def filterFunc(item:adc.Room):
    """
    CODE: GB 50016-2014 （2018年版）
    NAME: 建筑设计防火规范
    ITEM: 6.4.13
    CONTENT: 
    防火隔间的设置应符合下列规定：
    1 防火隔间的建筑面积不应小于6.0m2；
    """
    if ("防火隔间" in ",".join(item.vecAllName))and(item.RoomArea<6.0):
        return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% 未判断非正交
def filterFunc(item:adc.Room):
    """
    CODE: GB 50016-2014 （2018年版）
    NAME: 建筑设计防火规范
    ITEM: 7.3.5
    CONTENT: 
    消防电梯应设置前室，并应符合下列规定：
    2 前室的使用面积不应小于 6.0m2，前室的短边不应小于2.4m；"
    """
    if "消防电梯" in ",".join(item.vecAllName):
        for door in adc.getDescendants(item, adc.Door):
            nextRoom=adc.getNextRoom(item,door)
            maxRect=geom.getKthMaxRect(nextRoom.Profile["Outline"],nextRoom.Profile["Holes"])[0].bounds
            if ("前室" not in ",".join(nextRoom.vecAllName))\
                or(nextRoom.RoomArea<6.0)\
                or(maxRect[2]-maxRect[0]<2400.0)\
                or(maxRect[3]-maxRect[1]<2400.0):
                return False
    return True
checkers.append(Checker(adc.Room,filterFunc))

#%% 阳台尺寸需要修正
def filterFunc(item:adc.Family):
    """
    CODE: 
    NAME: 武汉市建设工程建筑面积计算规则
    ITEM: 3.3.2
    CONTENT: 
    未封闭阳台，水平投影面积总和不超过单套套内建筑面积的12%，且进深不超过2.4米
    """
    if item.SumBalconyFull+item.SumBalconyHalf*2>item.Area*0.12:
        return False
    for room in item.Rooms:
        if "阳台" in ",".join(room.vecAllName)\
            and 不封闭\
            and 进深>2400:
            return False
    return True
checkers.append(Checker(adc.Family,filterFunc))










#%% OK
def filterFunc(item:adc.Room):
    """建筑设计防火规范	6.4.3
    防烟楼梯间应符合下列规定：
    3 前室的使用面积：住宅建筑，不应小于4.5m2。
    与消防电梯间前室合用时，合用前室的使用面积：住宅建筑，不应小于6.0m2。
    """
    if "前室" not in ",".join(item.vecAllName): return True
    if item.RoomArea<4.5: return False
    for door in adc.getDescendants(item, adc.Door):
        nextRoom=item.getNextRoom(door)
        if (nextRoom is not None)and\
                ("消防电梯" in ",".join(nextRoom.vecAllName))and\
                (item.RoomArea<6.0):
            return False
    return True
checkers.append(Checker(adc.Room,filterFunc))



#%% 设备平台面积、尺寸需要修正
def filterFunc(item:adc.Family):
    """武汉市建设工程建筑面积计算规则	3.3.4
    1、住宅每个空调室外机搁板，不应超过0.8米，水平投影面积不应超过1.0平方米，且每套住宅空调室外机搁板水平投影面积总和应小于4.0平方米
    2、每套住宅独立设置的集中设备平台（包括集中空调室外机搁板），水平投影面积不应超过3.0平方米
    """
    minEdge,maxArea,sumArea,count=MAX_NUM,0,0,0
    for room in item.Rooms:
        if ("设备平台" in ",".join(room.vecAllName))or("空调" in ",".join(room.vecAllName)):
            sumArea+=room.RoomArea
            maxArea=max(maxArea,room.RoomArea)
            minEdge=min(minEdge,room.NetDepth,room.NetWidth)
            count+=1
    if (count==1)and(sumArea>3.0): return False
    if (count>1)and((minEdge>800)or(maxArea>1.0)or(sumArea>4.0)): return False
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% OK
def filterFunc(item:adc.Building):
    """建筑设计防火规范	5.5.3
    建筑的楼梯间宜通至屋面，通向屋面的门或窗应向外开启。
    """
    for room in adc.getDescendants(item,adc.Room):
        if ("楼梯" in ",".join(room.vecAllName))or(room.Stair is not None):
            for door in adc.getDescendants(room,adc.Door):
                if "屋面" in ",".join(room.getNextRoom(door).vecAllName):
                    return True
    return False
checkers.append(Checker(adc.Building,filterFunc))
#%% OK
def filterFunc(item:adc.Room):
    """住宅设计规范 GB50096-2011	6.2.6
    通向平屋面的门应向屋面方向开启。
    """
    if ("屋面" in ",".join(item.vecAllName)):
        for door in adc.getDescendants(item,adc.Door):
            if door.AxisReverse["Y"]:
                return False
    return True
checkers.append(Checker(adc.Room,filterFunc))
#%% OK
def filterFunc(item:adc.Room):
    """住宅设计规范 GB50096-2011	6.2.5
    楼梯间及前室的门应向疏散方向开启。
    """
    if ("楼梯" in ",".join(item.vecAllName))or("前室" in ",".join(item.vecAllName)):
        for door in adc.getDescendants(item,adc.Door):
            if "平开" not in door.StyleName:continue
            fromRoom,toRoom=door.getAdjacentRooms()
            if toRoom is None:continue
            if fromRoom is None or fromRoom.getPriority()<toRoom.getPriority():
                return False
    return True
checkers.append(Checker(adc.Room,filterFunc))
#%% 忽略：防火门等级
def filterFunc(item:adc.Family):
    """住宅设计规范 GB50096-2011	6.9.6
    地下楼、电梯间入口处应设置乙级防火门
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 忽略：防火门等级
def filterFunc(item:adc.Family):
    """民用建筑设计统一标准	8.3.2
    变电所防火门的级别应符合下列规定：
    1 变电所直接通向疏散走道(安全出口)的疏散门，以及变电所直接通向非变电所区域的门，应为甲级防火门；
    2 变电所直接通向室外的疏散门，应为不低于丙级的防火门。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 忽略：防火门等级
def filterFunc(item:adc.Family):
    """住宅建筑规范 GB50368-2005	9.4.3
    住宅建筑中竖井的设置应符合下列要求：
    4 电缆井和管道井设置在防烟楼梯间前室、合用前室时，其井壁上的检查门应采用丙级防火门。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 忽略：防火门等级
def filterFunc(item:adc.Family):
    """建筑设计防火规范	6.4.13
    防火隔间的设置应符合下列规定：2 防火隔间的门应采用甲级防火门；
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 忽略：防火门等级
def filterFunc(item:adc.Family):
    """建筑设计防火规范	6.4.5
    室外疏散楼梯应符合下列规定：
    4 通向室外楼梯的门应采用乙级防火门，并应向外开启。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 建筑高度、楼梯间拼装待完善
def filterFunc(item:adc.Staircase):
    """建筑防烟排烟系统技术标准	3.2.1
    当建筑高度大于10m时，尚应在楼梯间的外墙上每5层内设置总面积不小于2．0m2的可开启外窗或开口。
    """
    if "楼梯" not in ",".join(item.vecAllName): return True
    if adc.getAncestor(item,adc.Building).RoofHeight<10.0+1e-6:
            return True
checkers.append(Checker(adc.Staircase,filterFunc))
#%% 
def filterFunc(item:adc.Room):
    """建筑防烟排烟系统技术标准	3.2.2
    前室采用自然通风方式时，独立前室、消防电梯前室可开启外窗或开口的面积不应小于2．0m2，共用前室、合用前室不应小于3．0m2。
    """
    if "前室" not in ",".join(item.vecAllName): return True
    
    return True
checkers.append(Checker(adc.Room,filterFunc))
#%% 
def filterFunc(item:adc.Family):
    """建筑防烟排烟系统技术标准	3.3.11
    靠外墙的防烟楼梯间，尚应在其外墙上每5层内设置总面积不小于2m2的固定窗。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 
def filterFunc(item:adc.Family):
    """建筑设计防火规范	6.4.2
    封闭楼梯间应符合下列规定：
    2 除楼梯间的出入口和外窗外，楼梯间的墙上不应开设其他门、窗、洞口。
    3 封闭楼梯间的门应采用乙级防火门，并应向疏散方向开启；
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 
def filterFunc(item:adc.Family):
    """建筑设计防火规范	7.3.5
    消防电梯应符合下列规定：
    3除前室的出入口、前室内设置的正压送风口和本规范第 5.5.27 条规定的户门外，前室内不应开设其他门、窗、洞口；
    4前室或合用前室的门应采用乙级防火门。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 
def filterFunc(item:adc.Family):
    """建筑设计防火规范	6.4.3
    防烟楼梯间应符合下列规定：
    4 疏散走道通向前室以及前室通向楼梯间的门应采用乙级防火门。
    5 除住宅建筑的楼梯间前室外，防烟楼梯间和前室内的墙上不应开设除疏散门和送风口外的其他门、窗、洞口。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%% 
def filterFunc(item:adc.Family):
    """建筑设计防火规范	5.5.30
    住宅建筑的户门和安全出口的净宽度不应小于0.90m，首层疏散外门的净宽度不应小于1.10m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范 GB50096-2011	6.2.1
    十层以下的住宅建筑，当住宅单元任一层的建筑面积大于650m2，该住宅单元每层的安全出口不应少于2个。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.2.2
    如果住宅地上建筑层数满足十层及十层以上且不超过十八层并满足以下条件之一时：
    1.住宅单元最大楼层面积大于650m2，
    2.那么该住宅单元每层的安全出口不应少于2个。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.2.3
    十九层及十九层以上的住宅建筑，每层住宅单元的安全出口不应少于2个。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	5.5.26
    建筑高度大于27m，但不大于54m的住宅建筑，每个单元设置一座疏散楼梯时，疏散楼梯应通至屋面，且单元之间的疏散楼梯应能通过屋面连通，户门应采用乙级防火门。
    当不能通至屋面或不能通过屋面连通时，应设置2个安全出口。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.2.4
    安全出口应分散布置，两个安全出口的距离不应小于5m。    
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	5.5.2
    建筑内的安全出口和疏散门应分散布置，且建筑内每个防火分区或一个防火分区的每个楼层、每个住宅单元每层相邻两个安全出口以及每个房间相邻两个疏散门最近边缘之间的水平距离不应小于5m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅建筑规范GB 50368-2005	9.5.3
    在楼梯间的首层应设置直接对外的出口，或将对外出口设置在距离楼梯间不超过15m处。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	5.5.25
    住宅建筑安全出口的设置应符合下列规定：
    1 建筑高度不大于27m的建筑，当每个单元任一层的建筑面积大于650m2，或任一户门至最近安全出口的距离大于15m时，每个单元每层的安全出口不应少于2个；
    2 建筑高度大于27m、不大于54m的建筑，当每个单元任一层的建筑面积大于650m2，或任一户门至最近安全出口的距离大于10m时，每个单元每层的安全出口不应少于2个；
    3 建筑高度大于54m的建筑，每个单元每层的安全出口不应少于2个。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	7.3.5
    消防电梯应符合下列规定：
    1 前室宜靠外墙设置，并应在首层直通室外或经过长度不大于 30m 的通道通向室外；
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	5.5.27
    住宅建筑的疏散楼梯设置应符合下列规定：1 建筑高度不大于21m的住宅建筑可采用敞开楼梯间；与电梯井相邻布置的疏散楼梯应采用封闭楼梯间，当户门采用乙级防火门时，仍可采用敞开楼梯间。
    2 建筑高度大于21m、不大于33m的住宅建筑应采用封闭楼梯间；当户门采用乙级防火门时，可采用敞开楼梯间。
    3 建筑高度大于33m的住宅建筑应采用防烟楼梯间。户门不宜直接开向前室，确有困难时，每层开向同一前室的户门不应大于3樘且应采用乙级防火门。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	7.1.3
    卧室、起居室（厅）、厨房应有直接天然采光。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	7.1.5
    卧室、起居室(厅)、厨房的采光窗洞口的窗地面积比不应低于1／7。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	7.1.6
    当楼梯间设置采光窗时，采光窗洞口的窗地面积比不应低于1/12。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """被动式超低能耗居住建筑节能设计规范	5.2.4
    在兼顾保温隔热基础上保证立面采光窗的设置面积，应保证主要功能房间窗地面积比不低于1/6
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	7.2.1
    卧室、起居室（厅）、厨房应有自然通风。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	6.4.1
    疏散楼梯间应符合下列规定：
    1 楼梯间应能天然采光和自然通风，并宜靠外墙设置。靠外墙设置时，楼梯间、前室及合用前室外墙上的窗口与两侧门、窗、洞口最近边缘的水平距离不应小于1.0m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	5.1.1
    住宅应按套型设计，每套住宅应设卧室、起居室(厅)、厨房和卫生间等基本功能空间。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	5.4.4
    卫生间不应直接布置在下层住户的卧室、起居室(厅)、厨房和餐厅的上层。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.9.1
    卧室、起居室（厅）、厨房不应布置在地下室
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅建筑规范GB 50368-2005	5.2.6
    住宅建筑中设有管理人员室时，应设管理人员使用的卫生间。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.4.7
    电梯不应紧邻卧室布置。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	7.3.5
    起居室（厅）不宜紧邻电梯布置。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅建筑规范GB 50368-2005	7.1.5
    电梯不应与卧室、起居室紧邻布置。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """无障碍设计规范	8.1.4
    建筑内设有电梯时，至少应设置 1 部无障碍电梯。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.4.2
    十二层及十二层以上的住宅，每栋楼设置电梯不应少于两台，其中应设置一台可容纳担架的电梯。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅建筑规范GB 50368-2005	9.8.3
    12层及12层以上的住宅应设置消防电梯。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.4.1
    1.七层及七层以上住宅必须设置电梯；
    2.住户入口层楼面距室外设计地面（室外地坪）的高度超过16m时必须设置电梯；
    3.底层（首层）作为商店或其他用房的六层及六层以下住宅，且其住户入口层楼面距该建筑物的室外设计地面（室外地坪）高度超过16m时必须设置电梯；
    4.底层（首层）做架空层或贮存空间的六层及六层以下住宅，且其住户入口层楼面距该建筑物的室外设计地面（室外地坪）高度超过16m时必须设置电梯；
    5.顶层为两层一套的跃层住宅时，跃层部分不计层数，且其顶层住户入口层楼面距该建筑物室外设计地面（室外地坪）的高度超过16m时必须设置电梯。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.4.5
    七层及七层以上住宅电梯应在设有户门和公共走廊的每层设站。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅电梯配置和选型通用要求--武汉市	
    大于32层住宅，四户平面应配置3部电梯（规范层高按3米考虑）
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	8.1.6
    消防水泵房的设置应符合下列规定：
    2 附设在建筑内的消防水泵房，不应设置在地下三层及以下或室内地面与室外出入口地坪高差大于10m的地下楼层；"
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """民用建筑电气设计标准	4.2.1
    变电所位置选择，应符合下列要求：
    6 不应设在厕所、浴室、厨房或其他经常有水并可能漏水场所的正下方，且不宜与上述场所贴邻；如果贴邻，相邻隔墙应做无渗漏、无结露等防水处理；
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.3.1
    楼梯梯段净宽不应小于1.10m，不超过六层的住宅，一边设有栏杆的梯段净宽不应小于1.00m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	5.7.3
    套内楼梯当一边临空时，梯段净宽不应小于0.75m；当两侧有墙时，墙面之间净宽不应小于0.90m，并应在其中一侧墙面设置扶手。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.3.3
    楼梯平台净宽不应小于楼梯梯段净宽，且不得小于1.20m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """住宅设计规范GB 50096-2011	6.3.4
    楼梯为剪刀梯时，楼梯平台的净宽不得小于1.30m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """民用建筑空间规范	5.3.5
    梯段改变方向时，楼梯休息平台的最小宽度不应小于梯段净宽（扶手转向端处休息 平台净宽从踏步边缘算起），并不得小于 1．20m；当中间有实体墙时扶手转向端处的平台 净宽不应小于 1．30m。直跑楼梯的中间平台宽度不应小于 0．90m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """民用建筑设计统一标准	6.8.5
    每个梯段的踏步级数不应少于3级，且不应超过18级。    
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	6.4.5
    室外疏散楼梯应符合下列规定：
    1 栏杆扶手的高度不应小于1.10m，楼梯的净宽度不应小于0.90m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """建筑设计防火规范	5.5.30
    疏散楼梯的净宽度不应小于1.10m。建筑高度不大于18m的住宅中一边设置栏杆的疏散楼梯，其净宽度不应小于1.0m。
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """无障碍设计规范	3.5.3
    门的无障碍设计应符合下列规定:
    3 平开门开启后的通行净宽度不应小于 800mm；
    4 在门扇内外应留有直径不小于 1.50m 的轮椅回转空间；
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))
#%%
def filterFunc(item:adc.Family):
    """暖通需求	
    风井最小土建净宽尺寸为550mm
    """
    return True
checkers.append(Checker(adc.Family,filterFunc))




#%% 输出
#%% 
for i in checkers:
    i.check(dataBuilding)
    if len(i.failedObj)>0: 
        print(checkers.index(i),i.filterFunc.__doc__)