import math
from lib.linalg import Vec3d,Mat3d
from lib.geom import Node,Arc,LineSeg

#%% 椭圆-直线求交测试
if __name__=="__main__":
    s=Vec3d(1000,0)
    e=Vec3d(0,1500)
    c=Vec3d(500,300)

    rx,ry=500,250
    vx=Vec3d(2,0,0).rotate2d(math.pi/4)
    vy=Vec3d(0,1,0).rotate2d(math.pi/4)
    vz=Vec3d(0,0,1)

    basis=Mat3d.from_column_vecs([vx,vy,vz])
    basis_inv=basis.inverse()

    s2=basis_inv@s
    e2=basis_inv@e
    c2=basis_inv@c
    r2=250

    cir2=Arc.from_center_radius_angle(Node.from_vec3d(c2),r2,0,math.pi)
    line2=LineSeg(Node.from_vec3d(s2),Node.from_vec3d(e2))

    intersection=LineSeg.intersection_of_circle_and_line(cir2,line2)
    print(intersection)

    for pt2 in intersection:
        pt=basis@(pt2.to_vec3d())
        print(pt)

