from lib.linalg import Vec3d,Mat3d
class CADEntity:
    def __init__(self,object_name:str,layer:str,color:int) -> None:
        self.object_name=object_name
        self.layer=layer
        self.color=color
    @classmethod
    def parse(cls,ent)->"CADEntity":
        ent_type=_ENT_CLASS_MAP.get(ent.ObjectName)
        if ent_type is None: return 
        return ent_type(ent)
class CADPoint(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("point",ent.Layer,ent.Color)
        self.point=ent.Coordinates[:]
class CADLine(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("line",ent.Layer,ent.Color)
        self.start_point=ent.StartPoint[:]
        self.end_point=ent.EndPoint[:]
class CADArc(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("arc", ent.Layer, ent.Color)
        self.center=ent.Center[:]
        self.start_angle=ent.StartAngle
        self.end_angle=ent.EndAngle
        self.start_point=ent.StartPoint[:]
        self.end_point=ent.EndPoint[:]
        self.radius=ent.Radius
        self.normal=ent.Normal[:]
class CADPolyline(CADEntity):
    class _Segment:
        def __init__(self,ent,coords,i) -> None:
            self.start_width=0.0
            self.end_width=0.0     
            self.start_width,self.end_width=ent.GetWidth(i,self.start_width,self.end_width)       
            self.bulge=ent.GetBulge(i)
            self.start_point=[coords[i * 2],coords[i * 2 + 1],0]
    def __init__(self,ent) -> None:
        super().__init__("polyline", ent.Layer, ent.Color)
        self.segments=[]
        self.is_closed=ent.Closed
        coords=ent.Coordinates
        for j in range(len(coords) // 2):
            self.segments.append(self._Segment(ent,coords,j))
class CADHatch(CADEntity):
    def __init__(self,ent) -> None:
        super().__init__("hatch", ent.Layer, ent.Color)
        self.pattern_name=ent.PatternName
        self.angle=ent.Angle
        self.scale=ent.Scale
        self.is_closed=ent.Closed
        self.is_visible=ent.Visible
        self.is_filled=ent.IsFilled
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
}