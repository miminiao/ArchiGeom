from lib.linalg import Tensor,Vec3d,Vec4d,Mat3d,Mat4d
from lib.geom import Geom,Node,LineSeg,Arc,Polyedge,Loop
class CADEntity:
    def __init__(self,object_name:str,layer:str,color:int) -> None:
        self.object_name=object_name
        self.layer=layer
        self.color=color
    def to_geom(self)->Geom:...
    @classmethod
    def parse(cls,ent)->"CADEntity|None":
        ent_type=_ENT_CLASS_MAP.get(ent.ObjectName)
        if ent_type is not None: 
            return ent_type(ent)
    @classmethod
    def get_dxf_data(cls,ent,code:int,data_type:type,index:int=0)->list[int|float|str]:
        """获取组码为code的第index (starting from 0)个的数据"""
        doc=ent.Document
        type_map={int:"useri1",float:"userr1",str:"users1"}
        if data_type==list[float]:
            dxf_data=[]
            command_gen=lambda var,num,index,code,handle: f'(setvar "{var}" ({num} (nth {index} (mapcar \'cdr (vl-remove-if-not \'(lambda(x) (= {code} (car x))) (entget (handent "{handle}")))))))'
            command="".join(
                command_gen("userr1","car",index,code,ent.Handle),
                command_gen("userr2","cadr",index,code,ent.Handle),
                '(princ) ',
            )
            doc.SendCommand(command)
            dxf_data=[doc.GetVariable("userr1"),doc.GetVariable("userr2"),0]
        elif data_type in type_map:
            command=f'(setvar "{type_map[data_type]}" (nth {index} (mapcar \'cdr (vl-remove-if-not \'(lambda(x) (= {code} (car x))) (entget (handent "{ent.Handle}"))))))(princ) '
            doc.SendCommand(command)
            dxf_data=doc.GetVariable(type_map[data_type])
        else : raise ValueError('Value of "data_type" must be one of int, float, str or list[float].')
        return dxf_data
        
class CADPoint(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("point",ent.Layer,ent.Color)
        self.point:list[float]=ent.Coordinates[:]
    def to_geom(self) -> Node:
        return Node(*self.point)
class CADLine(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("line",ent.Layer,ent.Color)
        self.start_point:list[float]=ent.StartPoint[:]
        self.end_point:list[float]=ent.EndPoint[:]
    def to_geom(self) -> LineSeg:
        return LineSeg(Node(*self.start_point), Node(*self.end_point))
class CADArc(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("arc", ent.Layer, ent.Color)
        self.center:list[float]=ent.Center[:]
        self.start_angle:float=ent.StartAngle
        self.end_angle:float=ent.EndAngle
        self.total_angle:float=ent.TotalAngle
        self.start_point:list[float]=ent.StartPoint[:]
        self.end_point:list[float]=ent.EndPoint[:]
        self.radius:float=ent.Radius
        self.normal:list[float]=ent.Normal[:]
    def to_geom(self) -> Arc:
        return Arc.from_center_radius_angle(Node(*self.center), self.radius, self.start_angle, self.total_angle)
class CADPolyline(CADEntity):
    class _CADPolylineSegment:
        def __init__(self,ent,i) -> None:
            coords=ent.Coordinates[:]
            l=len(coords)
            self.start_point:list[float]=[coords[i*2], coords[i*2+1], 0]
            self.end_point:list[float]=[coords[(i+1)*2-l], coords[(i+1)*2+1-l], 0]
            self.start_width, self.end_width=ent.GetWidth(i)
            self.bulge:float=ent.GetBulge(i)
    def __init__(self,ent) -> None:
        super().__init__("polyline", ent.Layer, ent.Color)
        self.segments:list[CADPolyline._CADPolylineSegment]=[]
        self.is_closed:bool=ent.Closed
        for i in range(len(ent.Coordinates)//2):
            self.segments.append(CADPolyline._CADPolylineSegment(ent,i))
    def to_geom(self) -> Polyedge|Loop:
        if self.is_closed:
            # return Loop([(seg.start_point,seg.bulge) for seg in self.segments])
            return Loop.from_nodes([Node(*seg.start_point) for seg in self.segments])  # TODO 替换Loop的构造方法
        else: 
            return Polyedge([(seg.start_point,seg.bulge) for seg in self.segments])
        
class CADHatch(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("hatch", ent.Layer, ent.Color)
        self.pattern_name:str=ent.PatternName
        self.pattern_angle:str=ent.PatternAngle
        self.pattern_scale:float=ent.PatternScale
        self.elevation:float=ent.Elevation
        self.normal:list[float]=ent.Normal[:]
        self.origin:list[float]=ent.Origin[:]
        self.loops:list[CADPolyline]=[]
        if ent.AssociativeHatch:
            for i in range(ent.NumberOfLoops):
                self.loops.append(CADPolyline(ent.GetLoopAt(i)[0]))
        else:...  # TODO 用GetLoopAt()拿不到，需要从dxf组码读
class CADText(CADEntity):
    """CAD单/多行文字，天正单行文字"""
    def __init__(self,ent) -> None:
        super().__init__("text", ent.Layer, ent.Color)
        match ent.ObjectName:
            case "AcDbText":
                self.text=ent.TextString
            case "AcDbMText":
                self.text=ent.TextString.replace("\\P","\n")
            case "TDbText":
                self.text=ent.Text
        self.height=ent.Height,
        bounding_box=ent.GetBoundingBox()
        self.insert_point=[
            (bounding_box[0][0]+bounding_box[1][0])/2,
            (bounding_box[0][1]+bounding_box[1][1])/2,
            0,
        ]
class CADBlockRef(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("block_ref", ent.Layer, ent.Color)
        self.block_name=ent.Name
        self.effective_name=ent.EffectiveName
        self.insert_point=ent.InsertionPoint[:]
        self.scale=[
            ent.XScaleFactor,
            ent.YScaleFactor,
            ent.ZScaleFactor,
        ]
        self.rotation=ent.Rotation
        self.normal=ent.Normal[:]
        self.is_dynamic=ent.IsDynamicBlock
        self.dynamic_properties={prop.PropertyName:prop.Value for prop in ent.GetDynamicBlockProperties()}
        self.has_attributes=ent.HasAttributes
        self.attributes={attr.TagString:attr.TextString for attr in ent.GetAttributes()}
        CADBlockDef.parse(self.block_name)
    @property
    def basis3d(self)->Mat3d:
        vz=Vec3d(*self.normal)
        vx=Vec3d.Z().cross(vz)
        vx=vx.unit() if not vx.is_zero(is_unit=True) else Vec3d.X()
        vy=vz.cross(vx)
        return Mat3d.from_column_vecs([vx,vy,vz])
    @property
    def mat4d(self)->Mat4d:
        mat3d=self.basis3d @ self.rotation_mat @ self.scale_mat
        return Mat4d.from_row_vecs([Vec4d(*mat3d[0],self.insert_point[0]),
                                    Vec4d(*mat3d[1],self.insert_point[1]),
                                    Vec4d(*mat3d[2],self.insert_point[2]),
                                    Vec4d.W(),])
    @property
    def scale_mat(self)->Mat3d:
        return Mat3d.from_column_vecs([
            Vec3d(self.scale[0],0,0),
            Vec3d(0,self.scale[1],0),
            Vec3d(0,0,self.scale[2]),
        ])
    @property
    def rotation_mat(self)->Mat3d:
        return Mat3d.from_column_vecs([
            Vec3d.X().rotate2d(self.rotation),
            Vec3d.Y().rotate2d(self.rotation),
            Vec3d.Z(),
        ])
    def to_geom(self)->Geom|Tensor:
        if self.block_name=="_Matrix4d":
            return self.mat4d
        elif self.block_name=="_Vector3d":
            return Vec3d(*self.scale)
        else: ...

class CADBlockDef:
    blocks={}
    _doc_blocks={}
    def __init__(self,blk) -> None:
        self.object_name="block_def"
        self.block_name=blk.Name
        self.entities=[]
        for ent in blk:
            self.entities.append(CADEntity.parse(ent))
    @classmethod
    def init_doc_block_table(cls,doc_blocks)->dict:
        block_num=doc_blocks.Count
        i=0
        while i<block_num:
            try:
                blk=doc_blocks.Item(i)
                cls._doc_blocks[blk.Name]=blk
                i+=1
            except:...
        return cls._doc_blocks
    @classmethod
    def parse(cls,block_name)->"CADBlockDef":
        if block_name in cls.blocks: return cls.blocks[block_name]
        blk=cls._doc_blocks[block_name]
        cls.blocks[block_name]=CADBlockDef(blk)
        return cls.blocks[block_name]
class TZWall(CADEntity):
    """天正墙"""
    def __init__(self,ent) -> None:
        super().__init__("tzwall", ent.Layer, ent.Color)
        self.left_width:float=ent.LeftWidth  # 左宽
        self.right_width:float=ent.RightWidth  # 右宽
        self.elevation:float=ent.Elevation  # 标高
        self.height:float=ent.Height  # 高度
        self.insulate:str=ent.Insulate  # 保温: "无" | "双侧" | "内侧" | "外侧"
        self.insu_thick:float=ent.InsuThick  # 保温厚度
        self.left_insu_thick:float=ent.LeftInsuThick  # 左保温厚度
        self.right_insu_thick:float=ent.RightInsuThick  # 右保温厚度
        self.style:str=ent.Style  # 材料: "钢筋砼" | "混凝土" | "砖" | "耐火砖" | "石材" | "毛石" | "填充墙" | "加气块" | "空心砖" | "石膏板"
        self.usage:str=ent.Usage  # 用途: "外墙" | "内墙" | "分户墙" | "虚墙" | "矮墙" | "卫生隔断"
        self.start_point,self.end_point=self.get_endpoints(ent)  # 起终点
        self.is_arc:bool=ent.IsArc=="弧墙"  # 是否弧墙: "直墙" | "弧墙"
        self.radius:float=ent.Radius  # 圆弧半径，对于直墙radius=0
        self.total_angle=CADEntity.get_dxf_data(ent,50,float) if self.is_arc else 0  # 圆弧总角度
    def get_endpoints(self,ent)->tuple[list[float]]:
        doc=ent.Document
        command_gen=lambda var,pos,num,handle: f'(setvar "{var}" ({num} (vlax-curve-get{pos}Point (handent "{handle}"))))'
        command="".join(command_gen("userr1","Start","car",ent.Handle),
                        command_gen("userr2","Start","cadr",ent.Handle),
                        command_gen("userr3","End","car",ent.Handle),
                        command_gen("userr4","End","cadr",ent.Handle),
                        "(princ) ",
        )
        doc.SendCommand(command)
        s=[doc.GetVariable('userr1'),doc.GetVariable('userr2'),0]
        e=[doc.GetVariable('userr3'),doc.GetVariable('userr4'),0]
        return list(s),list(e)
class TZOpening(CADEntity):
    """天正门窗洞"""
    def __init__(self,ent,object_name:str) -> None:
        super().__init__(object_name, ent.Layer, ent.Color)
        self.kind:str=ent.GetKind  # 类别: "普通门" | "普通窗" |"弧窗" | "洞"
        self.height:float=ent.Height  # 高度
        self.width:float=ent.Width  # 宽度
        self.door_line:int=ent.DoorLine  # 门口线: 0="无" | 1="开启侧" | 2="背开侧" | 3="双侧" | 4="居中"
        self.up_lever:bool=ent.UpLever=="是"  # 位于上层
        self.style2d:str=CADEntity.get_dxf_data(ent,1,str)  # 2d样式
        self.position=self.get_dxf_data(ent,10,list[float])  # 插入点
        self.angle=CADEntity.get_dxf_data(ent,50,float)  # 旋转角度
    @classmethod
    def classifier(cls,ent)->"TZOpening":
        kind_map={"普通门": TZDoor,"普通窗": TZWindow,"弧窗":TZWindow,"洞":TZHole,}
        return kind_map[ent.GetKind](ent)
class TZDoor(TZOpening):
    """天正门"""
    def __init__(self,ent) -> None:
        super().__init__(ent,"tzdoor")
        self.door_sill:float=ent.DoorSill  # 门槛高
        self.evacuation_type:str=ent.EvacuationType  # 疏散类别: "无" | "房间疏散门|户门" | "安全出口"
        self.sub_kind:str=ent.GetSubKind  # 类型: "普通门" | "甲级防火门" | "乙级防火门" | "丙级防火门" | "防火卷帘" | "人防门" | "隔断门" | "电梯门"
class TZWindow(TZOpening):        
    """天正窗"""
    def __init__(self,ent) -> None:
        super().__init__(ent,"tzwindow")
        self.win_sill:float=ent.WinSill  # 窗台高
        self.is_high:bool=ent.IsHigh=="是"  # 高窗
        self.sub_kind:str=ent.GetSubKind  # 类型: "普通窗" | "防火窗" | "弧窗"
        self.is_arc:bool=ent.GetKind=="弧窗"  # 是否弧窗
class TZHole(TZOpening):
    """天正洞"""
    def __init__(self,ent) -> None:
        super().__init__(ent,"tzhole")
        self.win_sill:float=ent.WinSill  # 窗台高
        self.line_offset_distance:float=ent.LineOffsetDistance  # 偏移距离

_ENT_CLASS_MAP = {
    "AcDbPoint": CADPoint,
    "AcDbLine": CADLine,
    "AcDbArc": CADArc,
    "AcDbPolyline": CADPolyline,
    "AcDbHatch": CADHatch,
    "AcDbText" : CADText,
    "AcDbMText": CADText,
    "TDbText": CADText,
    "AcDbBlockReference": CADBlockRef,
    "AcDbHatch": CADHatch,
    "TDbWall": TZWall,
    "TDbOpening": TZOpening.classifier,
}