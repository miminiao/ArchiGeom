from abc import ABC,abstractmethod
import win32com.client
import pythoncom
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from shapely.geometry import Polygon as shPolygon
from lib.geom import Geom,Node,LineSeg,Arc,Edge,Polyedge,Loop,Polygon

class GeomPlotter(ABC):
    @classmethod
    @abstractmethod
    def draw_geoms(cls,geoms:list[Geom],show:bool=False,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_edge(cls,edge:Edge,show:bool=False,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_polyedge(cls,polyedge:Polyedge,show_node_text:bool=False,show:bool=False,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_loop(cls,loop:Loop,show_node:bool=False,show_text:bool=False,show:bool=False,*args,**kwargs)->None: ...
    @classmethod
    @abstractmethod
    def _draw_polygon(cls,poly: shPolygon | Polygon, show:bool=False, *args, **kwargs)->None: ...

class MPLPlotterr(GeomPlotter):
    @classmethod
    def draw_geoms(cls,geoms:list[Geom],show:bool=False,*args,**kwargs)->None:
        draw_method_dict={
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
        line_style="solid" if loop.area>0 else "dashed"
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
    def _draw_polygon(cls,poly: shPolygon | Polygon, show:bool=False, *args, **kwargs):
        x, y = poly.exterior.xy
        plt.plot(x, y, *args, **kwargs)
        for hole in poly.interiors:
            x, y = hole.xy
            plt.plot(x, y, *args, **kwargs)

class CADPlotter(GeomPlotter):
    model_space=None
    @classmethod
    def get_current_modelspace(cls):
        acad = win32com.client.Dispatch("AutoCAD.Application.23")
        doc = acad.ActiveDocument
        cls.model_space= doc.ModelSpace
    @classmethod
    def draw_geoms(cls,geoms:list[Geom],*args,**kwargs)->None:
        draw_method_dict={
            Node:       cls._draw_node,
            LineSeg:    cls._draw_edge,
            Arc:        cls._draw_edge,
            Polyedge:   cls._draw_polyedge,
            Loop:       cls._draw_loop,
            Polygon:    cls._draw_polygon,
        }
        cls.get_current_modelspace()
        for i,geom in enumerate(geoms):
            ent=draw_method_dict[type(geom)](geom,*args,**kwargs)
            if "color" in kwargs: ent.Color=kwargs["color"]
    @classmethod
    def _draw_node(cls,node:Node,*args,**kwargs):
        point=cls._com_point(node)
        ent=cls.model_space.AddPoint(point)
        return ent
    @classmethod
    def _draw_edge(cls,edge:Edge,*args,**kwargs):
        if isinstance(edge,LineSeg):
            s,e = cls._com_point(edge.s), cls._com_point(edge.e)
            ent=cls.model_space.AddLine(s, e)
        elif isinstance(edge,Arc):
            center=cls._com_point(edge.center)
            ent=cls.model_space.AddArc(center,edge.radius,*edge.angles)
        return ent
    @classmethod
    def _draw_polyedge(cls,polyedge:Polyedge, *args, **kwargs):
        end_points=cls._com_point_list(polyedge.nodes)
        ent=cls.model_space.AddPolyline(end_points)
        show_node_text=kwargs.get("show_node_text",False)
        if show_node_text:
            for i,node in enumerate(polyedge.nodes):
                cls.model_space.AddText(f"{i}",cls._com_point(node),250)
        return ent
    @classmethod
    def _draw_loop(cls,loop:Loop, *args, **kwargs):
        end_points=cls._com_point_list(loop.nodes)
        ent=cls.model_space.AddPolyline(end_points)
        ent.Closed=True
        ent.Color=1 if loop.area>=0 else 2
        show_node_text=kwargs.get("show_node_text",False)
        if show_node_text:
            for i,node in enumerate(loop.nodes):
                cls._draw_text(f"{id(loop)}:{i}",node,250)
        return ent
    @classmethod
    def _draw_text(cls,text:str,pos:Node,height:float,*args,**kwargs):
        point=cls._com_point(pos)
        ent=cls.model_space.AddText(text,point,height)
        return ent
    @staticmethod
    def _com_point(node:Node)->win32com.client.VARIANT:
        return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [node.x, node.y, node.z])
    @staticmethod
    def _com_point_list(nodes:list[Node])->win32com.client.VARIANT:
        coords=[]
        for node in nodes: coords.extend([node.x,node.y,node.z])
        return win32com.client.VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, coords)
