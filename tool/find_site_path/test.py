from shapely.geometry import Polygon
from centerline.geometry import Centerline
import matplotlib.pyplot as plt
from tool.find_site_path.plot_helper  import plot_polygon

polygon = Polygon([[0, 0], [0, 4], [30, 4], [30, 0]])
fig, ax=plt.subplots()
plot_polygon(ax,polygon)

attributes = {"id": 1, "name": "polygon", "valid": True}

centerline = Centerline(polygon, **attributes)

for l in centerline.geometry.geoms:
    plt.plot(*(l.xy))

plt.show()

