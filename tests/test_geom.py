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

