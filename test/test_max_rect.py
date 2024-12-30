from shapely.geometry import box as shBox
from shapely.affinity import rotate
from random import random, seed
from matplotlib import pyplot as plt

const=Constant.default()

fig = plt.figure(figsize=(10, 10))
ax = plt.gca()
ax.set_aspect(1)

# 随机多边形


seed(4)
SCALE = 100000.0
SHELL_BOX = 50
HOLE_BOX = 5
MAX_ROTATION = 90
while True:
    rand_poly = shBox(0, 0, 0, 0)
    for i in range(SHELL_BOX):  # SHELL
        new_box = rotate(
            shBox(
                random() * SCALE,
                random() * SCALE,
                random() * SCALE,
                random() * SCALE,
            ),
            random() * MAX_ROTATION,
        )
        rand_poly = rand_poly.union(new_box)
    for i in range(HOLE_BOX):  # HOLES
        new_box = rotate(
            shBox(
                random() * SCALE,
                random() * SCALE,
                random() * SCALE,
                random() * SCALE,
            ),
            random() * MAX_ROTATION,
        )
        rand_poly = rand_poly.difference(new_box)
    if isinstance(rand_poly, shPolygon):
        rand_poly = rand_poly.simplify(tolerance=const.TOL_DIST)
        break
exterior = Loop.from_nodes([Node(x, y) for x, y in rand_poly.exterior.coords])
interiors = [
    Loop.from_nodes([Node(x, y) for x, y in hole.coords])
    for hole in rand_poly.interiors
]
poly = Polygon(exterior, interiors)

# 包含点
ORDER = 1
POINT_NUM = 2
# covered_points = [[Node(random() * SCALE, random() * SCALE) for i in range(POINT_NUM)] for j in range(ORDER)]
covered_points = [[Node(60000, 10000), Node(50000, 15000)]]

# 求最大矩形
PRECISION = -1
CUT_DEPTH = 1

ins = MaxRectAlgo(
    poly=poly,
    order=ORDER,
    covered_points=covered_points,
    precision=PRECISION,
    cut_depth=CUT_DEPTH,
)
rects = ins.get_result()

_draw_polygon(rand_poly, color="r")
for i in range(len(rects)):
    if rects[i] is None:
        continue
    # print(i,rects[i].area)
    x, y = rects[i].xy
    plt.fill(x, y, color="b", alpha=0.8 - (0.8 / ORDER) * i)
for k in range(len(covered_points)):
    for l in range(len(covered_points[k])):
        plt.scatter(
            covered_points[k][l].x,
            covered_points[k][l].y,
            color="r",
            alpha=0.8 - (0.8 / ORDER) * k,
        )
plt.show()
