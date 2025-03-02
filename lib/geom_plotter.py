from abc import ABC,abstractmethod
import win32com.client
import pythoncom
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from lib.geom import Geom,Node,LineSeg,Arc,Edge,Polyedge,Loop,Polygon
from typing import Callable
from time import time

class GeomPlotter(ABC):
    @classmethod
    @abstractmethod
    def draw_geoms(cls,geoms:list[Geom],*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_node(cls,node:Node,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_edge(cls,edge:Edge,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_polyedge(cls,polyedge:Polyedge,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_loop(cls,loop:Loop,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_polygon(cls,poly:Polygon,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_text(cls,text:str,pos:Node,*args,**kwargs)->None: ...

class MPLPlotter(GeomPlotter):
    @classmethod
    def draw_geoms(cls,geoms:list[Geom],show:bool=False,*args,**kwargs)->None:
        draw_method_dict={
            Node:       cls._draw_node,
            LineSeg:    cls._draw_edge,
            Arc:        cls._draw_edge,
            Polyedge:   cls._draw_polyedge,
            Loop:       cls._draw_loop,
            Polygon:    cls._draw_polygon,
        }
        colors=list(TABLEAU_COLORS)
        if isinstance(geoms,Geom): geoms=[geoms]
        for i,geom in enumerate(geoms):
            kwargs["color"]=colors[i % len(colors)]
            draw_method_dict[type(geom)](geom,*args,**kwargs)
        if show:
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()
    @classmethod
    def _draw_node(cls,node:Node,show:bool=False,node_text:Callable[[Node],str]=None,*args,**kwargs):
        plt.scatter(node.x,node.y,*args,**kwargs)
        if node_text is not None:
            plt.text(node.x,node.y,node_text(node),color="b")
        if show:
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()
    @classmethod
    def _draw_edge(cls,edge:Edge,show:bool=False,*args,**kwargs):
        if isinstance(edge,LineSeg):
            plt.plot(*edge.to_array().T,*args,**kwargs)
        elif isinstance(edge,Arc):
            sub_edges=edge.fit()
            for sub_edge in sub_edges:
                plt.plot(sub_edge.to_array()[:,0], sub_edge.to_array()[:,1],*args,**kwargs)
        if show:
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()
    @classmethod
    def _draw_polyedge(cls,polyedge:Polyedge,show_node_text:bool=False,show:bool=False,*args,**kwargs)->None:
        for i,edge in enumerate(polyedge.edges):
            if isinstance(edge,LineSeg):
                plt.plot(*edge.to_array().T,*args,**kwargs)
            elif isinstance(edge,Arc):
                sub_edges=edge.fit()
                for sub_edge in sub_edges:
                    plt.plot(sub_edge.to_array()[:,0], sub_edge.to_array()[:,1],*args,**kwargs)
            if show_node_text:
                plt.scatter(edge.s.x,edge.s.y)
                plt.text(edge.s.x+1.0*i,edge.s.y+1.0*i,i,color="b")
                plt.plot([edge.s.x,edge.s.x+1.0*i],[edge.s.y,edge.s.y+1.0*i],alpha=0.1)
                if i==len(polyedge)-1:
                    plt.scatter(edge.e.x,edge.e.y)
                    plt.text(edge.e.x+1.0*i+1,edge.e.y+1.0*i+1,i+1,color="b")
                    plt.plot([edge.e.x,edge.e.x+1.0*i],[edge.e.y,edge.e.y+1.0*i],alpha=0.1)
        if show:
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()
    @classmethod
    def _draw_loop(cls,loop:Loop,show_node:bool=False,show_text:bool=False,show:bool=False,*args,**kwargs)->None:
        line_style="solid" if loop.area()>0 else "dashed"
        for i,edge in enumerate(loop.edges):
            if isinstance(edge,LineSeg):
                plt.plot(*edge.to_array().T,linestyle=line_style,*args,**kwargs)
            elif isinstance(edge,Arc):
                sub_edges=edge.fit()
                for sub_edge in sub_edges:
                    plt.plot(sub_edge.to_array()[:,0], sub_edge.to_array()[:,1],linestyle=line_style,*args,**kwargs)
            if show_node:
                plt.scatter(edge.s.x,edge.s.y)
                if show_text:
                    plt.text(edge.s.x+1.0*i,edge.s.y+1.0*i,f"{i}.{i}",color="b")
                    plt.plot([edge.s.x,edge.s.x+1.0*i],[edge.s.y,edge.s.y+1.0*i],alpha=0.1)
        if show:
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()
    @classmethod
    def _draw_polygon(cls,poly:Polygon, show:bool=False, *args, **kwargs):
        for loop in poly.all_loops: cls._draw_loop(loop,show=show,*args,**kwargs)
        # plt.fill
    @classmethod
    def _draw_text(cls,text:str,pos:Node,*args,**kwargs)->None:
        plt.text(pos.x,pos.y,text)
class CADPlotter(GeomPlotter):
    _model_space=None
    _blocks=None
    @classmethod
    def _get_current_doc(cls):
        acad = win32com.client.Dispatch("AutoCAD.Application.23")
        doc = acad.ActiveDocument
        cls._model_space= doc.ModelSpace
        cls._blocks=doc.Blocks
    @classmethod
    def draw_geoms(cls,geoms:Geom|list[Geom],*args,**kwargs)->None:
        draw_method_dict={
            Node:       cls._draw_node,
            LineSeg:    cls._draw_edge,
            Arc:        cls._draw_edge,
            Polyedge:   cls._draw_polyedge,
            Loop:       cls._draw_loop,
            Polygon:    cls._draw_polygon,
        }
        cls._get_current_doc()
        if isinstance(geoms,Geom): geoms=[geoms]
        for i,geom in enumerate(geoms):
            ent=draw_method_dict[type(geom)](cls._model_space,geom,*args,**kwargs)
            if "color" in kwargs: ent.Color=kwargs["color"]
    @classmethod
    def _draw_node(cls,current_space,node:Node,node_text:Callable[[Node],str]=None,*args,**kwargs):
        point=cls._point_to_com(node)
        ent=current_space.AddPoint(point)
        if "color" in kwargs: ent.Color=kwargs["color"]
        if node_text is not None: 
            cls._draw_text(node_text(node),node)
        return ent
    @classmethod
    def _draw_edge(cls,current_space,edge:Edge,*args,**kwargs):
        if isinstance(edge,LineSeg):
            s,e = cls._point_to_com(edge.s), cls._point_to_com(edge.e)
            ent=current_space.AddLine(s, e)
        elif isinstance(edge,Arc):
            center=cls._point_to_com(edge.center)
            ent=current_space.AddArc(center,edge.radius,*edge.angles)
        return ent
    @classmethod
    def _draw_polyedge(cls,current_space,polyedge:Polyedge, *args, **kwargs):
        end_points=cls._point_list_to_com(polyedge.nodes)
        ent=current_space.AddPolyline(end_points)
        for i,bulge in enumerate(polyedge.bulges):
            ent.SetBulge(i,bulge)
        show_node_text=kwargs.get("show_node_text",False)
        if show_node_text:
            for i,node in enumerate(polyedge.nodes):
                current_space.AddText(f"{i}",cls._point_to_com(node),250)
        return ent
    @classmethod
    def _draw_loop(cls,current_space,loop:Loop, *args, **kwargs):
        end_points=cls._point_list_to_com(loop.nodes)
        ent=current_space.AddPolyline(end_points)
        for i,edge in enumerate(loop.edges):
            ent.SetBulge(i,edge.bulge if isinstance(edge,Arc) else 0)
        ent.Closed=True
        ent.Color=2 if loop.area>=0 else 1
        show_node_text=kwargs.get("show_node_text",False)
        if show_node_text:
            for i,node in enumerate(loop.nodes):
                cls._draw_text(f"{id(loop)}:{i}",node)
        return ent
    @classmethod
    def _draw_polygon(cls,current_space,polygon:Polygon, fill:bool=True, fill_opacity:int=80, *args, **kwargs):
        base_point=cls._point_to_com(Node(0,0,0))
        block_name=f"Polygon_{time()}"
        new_block=cls._blocks.Add(base_point,block_name)
        outer=cls._draw_loop(new_block,polygon.shell)
        inners=[]
        for loop in polygon.holes:
            inners.append(cls._draw_loop(new_block,loop))
        if fill:
            hatch=new_block.AddHatch(1,"SOLID",True,0)
            hatch.AppendOuterLoop(cls._obj_list_to_com([outer]))
            for inner in inners:
                hatch.AppendInnerLoop(cls._obj_list_to_com([inner]))
            hatch.EntityTransparency=fill_opacity
        insertion_point=cls._point_to_com(Node(0,0,0))
        ent=current_space.InsertBlock(insertion_point,block_name,1,1,1,0)
        return ent
    @classmethod
    def _draw_text(cls,current_space,text:str,pos:Node,height:float=250,*args,**kwargs):
        point=cls._point_to_com(pos)
        ent=current_space.AddText(text,point,height)
        return ent
    @staticmethod
    def _point_to_com(node:Node)->win32com.client.VARIANT:
        return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [node.x, node.y, node.z])
    @staticmethod
    def _point_list_to_com(nodes:list[Node])->win32com.client.VARIANT:
        coords=[]
        for node in nodes: coords.extend([node.x,node.y,node.z])
        return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, coords)
    @staticmethod
    def _obj_list_to_com(objs:list)->win32com.client.VARIANT:
        return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_DISPATCH, objs)
