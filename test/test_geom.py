#%% 自相交测试（手绘）
if 0 and __name__=="__main__":
    import tkinter as tk    
    from random import random
    from lib.geom_plotter import MPLPlotterr
    def add_points(window,canvas,pts,event):
        global ended
        if ended: window.destroy()
        r=5.0
        canvas.create_oval(event.x-r/2,event.y-r/2,event.x+r/2,event.y+r/2)
        if len(pts)>0:
            canvas.create_line(pts[-1][0],pts[-1][1],event.x,event.y)
        pts.append([event.x,event.y])
    def close_polyline(window,canvas,pts,event):
        global ended
        if ended: window.destroy()
        if len(pts)>0:
            canvas.create_line(pts[-1][0],pts[-1][1],pts[0][0],pts[0][1])
        ended=True

    pts=[]
    ended=False
    h,w=400,400
    window=tk.Tk()
    canvas = tk.Canvas(window,bg="#ffffff",height=h,width=w)  
    canvas.bind("<Button-1>",lambda event:add_points(window,canvas,pts,event))
    canvas.bind("<Button-3>",lambda event:close_polyline(window,canvas,pts,event))
    canvas.pack()
    window.mainloop()
    pts=np.array([[pt[0],h-pt[1]]for pt in pts])
    plt.figure()
    
    loop=Loop.from_array(pts)
    loops_spl=loop.split_self_intersection(True)
    
    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.set_aspect(1)
    MPLPlotterr.draw_geoms([shPolygon(pts)])

    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.set_aspect(1)    
    for l in loops_spl:
        _draw_polygon(shPolygon(l.to_array()),color=("C%d"%int(random()*10),"C%d"%int(random()*10)))

    plt.show()

#%% 自相交测试
if 0 and __name__=="__main__":
    import json
    from random import random
    with open("self_crossing.json") as f:
        polys=json.load(f)
    plt.figure()
    column_num=len(polys)
    for i,poly in enumerate(polys):    
        poly=np.array(poly)
        loop=Loop.from_array(poly)
        loops_spl=loop.split_self_intersection(positive=True,ensure_valid=False)
    
        plt.subplot(2,column_num,i+1)
        ax = plt.gca()
        ax.set_aspect(1)
        _draw_polygon(shPolygon(poly))

        plt.subplot(2,column_num,i+1+column_num)
        ax = plt.gca()
        ax.set_aspect(1)    
        for j,l in enumerate(loops_spl):
            _draw_polygon(shPolygon(l.to_array()),color=(f"C{j}",f"C{j}"))

    plt.show()

#%% 圆弧mbb测试
if 0 and __name__=="__main__":
    # edge2=Edge(Node(-189.1,-219.0),Node(-169.4,-234.9)).opposite()
    # edge1=Edge(Node(-160.4,-233.5),Node(-139.5,-214.0)).opposite()
    # plt.plot(edge1.to_array()[:,0], edge1.to_array()[:,1])
    # plt.plot(edge2.to_array()[:,0], edge2.to_array()[:,1])
    # arc=edge1.fillet_with(edge2,12)
    arc=Arc.from_center_radius_angle(Node(0,0),100,0.1,6)

    p1,p2=arc.s,arc.e
    plt.scatter([p1.x,p2.x,arc.center.x],[p1.y,p2.y,arc.center.y])
    edges=arc.fit()
    for edge in edges:
        plt.plot(edge.to_array()[:,0], edge.to_array()[:,1],c='b')
    mbb=arc.get_mbb()
    plt.plot([mbb[0].x,mbb[1].x,mbb[1].x,mbb[0].x,mbb[0].x],
             [mbb[0].y,mbb[0].y,mbb[1].y,mbb[1].y,mbb[0].y])
    ax = plt.gca()
    ax.set_aspect(1) 
    plt.show()

#%% fillet测试
if 0 and __name__=="__main__":
    import tkinter as tk    
    def add_points(window,canvas,pts,event):
        global ended
        if ended: window.destroy()
        r=5.0
        canvas.create_oval(event.x-r/2,event.y-r/2,event.x+r/2,event.y+r/2)
        if len(pts)>0:
            canvas.create_line(pts[-1][0],pts[-1][1],event.x,event.y)
        pts.append([event.x,event.y])
    def close_polyline(window,canvas,pts,event):
        global ended
        if ended: window.destroy()
        if len(pts)>0:
            canvas.create_line(pts[-1][0],pts[-1][1],pts[0][0],pts[0][1])
        ended=True

    pts=[]
    ended=False
    h,w=600,600
    window=tk.Tk()
    canvas = tk.Canvas(window,bg="#ffffff",height=h,width=w)  
    canvas.bind("<Button-1>",lambda event:add_points(window,canvas,pts,event))
    canvas.bind("<Button-3>",lambda event:close_polyline(window,canvas,pts,event))
    canvas.pack()
    window.mainloop()
    pts=np.array([[pt[0],h-pt[1]]for pt in pts])
    plt.figure()
    
    loop=Loop.from_array(pts)
    loop_fillet=loop.fillet(30,mode="relax")

    plt.subplot(1,2,1)
    ax = plt.gca()
    ax.set_aspect(1)
    _draw_polygon(shPolygon(pts))

    plt.subplot(1,2,2)
    ax = plt.gca()
    ax.set_aspect(1)    
    _draw_polygon(shPolygon(loop_fillet.to_array()))

    plt.show()

#%% 圆弧求交测试
if 0 and __name__=="__main__":
    arc1=Arc.from_center_radius_angle(Node(0,0),1200,0,math.pi*0.5)
    arc2=Arc.from_center_radius_angle(Node(1200,400),1000,math.pi*0.5,math.pi*0.5)
    inter=arc1.intersection(arc2)
    print(inter)
    
    edges=arc1.fit()
    for edge in edges:
        plt.plot(edge.to_array()[:,0], edge.to_array()[:,1],c='b')
    edges=arc2.fit()
    for edge in edges:
        plt.plot(edge.to_array()[:,0], edge.to_array()[:,1],c='b')        
    
    ax = plt.gca()
    ax.set_aspect(1) 
    plt.show()

#%% 椭圆-直线求交测试
if 0 and __name__=="__main__":
    s=Vec3d(1000,0)
    e=Vec3d(0,1500)
    c=Vec3d(500,300)

    rx,ry=500,250
    vx=Vec3d(2,0,0).rotate2d(math.pi/4)
    vy=Vec3d(0,1,0).rotate2d(math.pi/4)
    vz=Vec3d(0,0,1)
    from lib.linalg import Mat3d
    basis=Mat3d.from_column_vecs([vx,vy,vz])
    basis_inv=basis.invert()

    s2=basis_inv@s
    e2=basis_inv@e
    c2=basis_inv@c
    r2=250

    cir2=Arc.from_center_radius_angle(Node.from_vec3d(c2),r2,0,math.pi)
    line2=LineSeg(Node.from_vec3d(s2),Node.from_vec3d(e2))

    intersection=Edge.intersection_of_circle_and_line(cir2,line2)
    print(intersection)

    for pt2 in intersection:
        pt=basis@(pt2.to_vec3d())
        print(pt)

#%% Loop面积测试
if 1 and __name__=="__main__":
    import json
    from tool.converter.json_converter import cad_polyline_to_loop
    with open(f"test/split_loop/case_a.json",'r',encoding="utf8") as f:
        j_obj=json.load(f)
    loops=cad_polyline_to_loop(j_obj)
    print(loops[0].get_area())