from lib.linalg import *
ref_create_inv=Mat4d([
    [-1,0,0,3.01169e6],
    [0,1,0,-2.08387e6],
    [0,0,1,0],
    [0,0,0,1]
    ])
ref_create=ref_create_inv.inverse()

world2clip=Mat4d([
    [1,0,0,188341],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
    ])
clip2world=world2clip.inverse()
ref_now=Mat4d([
    [1,0,0,-1914113],
    [0,1,0,1553549],
    [0,0,1,0],
    [0,0,0,1]
    ])

v1_in_clip=Vec4d(1.46962e6,532788,0,1)
v2_in_clip=Vec4d(1.46962e6,532326,0,1)

import numpy
numpy.set_printoptions(suppress=True)

v1_create_global=clip2world@v1_in_clip
v1_create_local=ref_create_inv@v1_create_global
v1_now_local=v1_create_local
v1_now_global=ref_now@v1_now_local


v2_create_global=clip2world@v2_in_clip
v2_create_local=ref_create_inv@v2_create_global
v2_now_local=v2_create_local
v2_now_global=ref_now@v2_now_local

from lib.geom_plotter import CADPlotter
from lib.geom import LineSeg,Node
CADPlotter.draw_geoms([LineSeg(Node.from_vec3d(v1_now_global),Node.from_vec3d(v2_now_global))])

print(v1_now_global)
print(v2_now_global)