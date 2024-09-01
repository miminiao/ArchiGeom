import base64
from lib.linalg import Vec3d,Mat3d
class CADEntity:
    def __init__(self,object_name:str,layer:str,color:int) -> None:
        self.object_name=object_name
        self.layer=layer
        self.color=color
    @classmethod
    def parse(cls,ent)->"CADEntity|None":
        ent_type=_ENT_CLASS_MAP.get(ent.ObjectName)
        if ent_type is not None: 
            return ent_type(ent)
    @classmethod
    def get_dxf_data(cls,ent)->dict[int:str]:
        dxf_data={}
        doc=ent.Document
        doc.SendCommand(f'(setvar "users1" (vl-princ-to-string (entget (handent "{ent.Handle}"))))(princ) ')
        dxf_str=doc.GetVariable('users1')
        split_str=dxf_str.strip("()").split(') (')
        for s in split_str:
            k,v=s.split(' . ')
            k=int(k)
            if k not in dxf_data: dxf_data[k]=[]
            dxf_data[k].append(v)
        return dxf_data
        
class CADPoint(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("point",ent.Layer,ent.Color)
        self.point:list[float]=ent.Coordinates[:]
class CADLine(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("line",ent.Layer,ent.Color)
        self.start_point:list[float]=ent.StartPoint[:]
        self.end_point:list[float]=ent.EndPoint[:]
class CADArc(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("arc", ent.Layer, ent.Color)
        self.center:list[float]=ent.Center[:]
        self.start_angle:float=ent.StartAngle
        self.end_angle:float=ent.EndAngle
        self.start_point:list[float]=ent.StartPoint[:]
        self.end_point:list[float]=ent.EndPoint[:]
        self.radius:float=ent.Radius
        self.normal:list[float]=ent.Normal[:]
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
        self.insert_point=ent.InsertionPoint[:]
        self.scale=[
            ent.XScaleFactor,
            ent.YScaleFactor,
            ent.ZScaleFactor,
        ]
        self.rotation=ent.Rotation
        self.normal=ent.Normal[:]
        self.is_dynamic=ent.IsDynamicBlock
        self.block_name=ent.Name
        self.effective_name=ent.EffectiveName
        CADBlockDef.parse(self.block_name)
    def get_basis_mat(self)->Mat3d:
        vx=Vec3d(0,0,1).cross(self.normal)
        if vx.is_zero(is_unit=True): vx=Vec3d(1,0,0)
        vz=self.normal
        vy=self.normal.cross(vx)
        return Mat3d.from_column_vecs([vx,vy,vz])
    def get_scale_mat(self)->Mat3d:
        return Mat3d.from_column_vecs([
            Vec3d(self.scale[0],0,0),
            Vec3d(0,self.scale[1],0),
            Vec3d(0,0,self.scale[2]),
        ])
    def get_rotation_mat(self)->Mat3d:
        return Mat3d.from_column_vecs([
            Vec3d(1,0,0).rotate2d(self.rotation),
            Vec3d(0,1,0).rotate2d(self.rotation),
            Vec3d(0,0,1),
        ])
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
        self.left_width=ent.LeftWidth  # 左宽
        self.right_width=ent.RightWidth  # 右宽
        self.elevation=ent.Elevation  # 标高
        self.height=ent.Height  # 高度
        self.insulate=ent.Insulate  # 保温      
        self.insu_thick=ent.InsuThick  # 保温厚度
        self.left_insu_thick=ent.LeftInsuThick  # 左保温厚度
        self.right_insu_thick=ent.RightInsuThick  # 右保温厚度
        self.is_arc=ent.IsArc  # 是否弧墙: 直墙|弧墙
        self.radius=ent.Radius  # 弧半径
        self.style=ent.Style  # 材料: 钢筋砼|混凝土|砖|耐火砖|石材|毛石|填充墙|加气块|空心砖|石膏板
        self.usage=ent.Usage  # 用途: 外墙|内墙|分户墙|虚墙|矮墙|卫生隔断
        
        dxf_data=CADEntity.get_dxf_data(ent)
        baseline=base64.decode(dxf_data[300]).split(",")
        
    def get_height_vec(self)->Vec3d:
        return Vec3d(0,0,self.height)
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
}
