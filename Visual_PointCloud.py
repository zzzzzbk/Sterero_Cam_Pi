import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import pyvista as pv
import numpy as np
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select PLY file",
    filetypes=[("PLY files", "*.ply"), ("All files", "*.*")]
)
if not file_path:
    raise Exception("No file selected.")
# Load the point cloud from the PLY file
pcd = o3d.io.read_point_cloud(file_path)
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="depth_map", width=800, height=600)

# points=np.asarray(pcd.points)
# colors=pcd.colors
# cloud = pv.PolyData(points)
# cloud["z_color"] = colors

# # Create a plotter
# plotter = pv.Plotter()
# plotter.add_points(
#     cloud,
#     scalars="z_color",      # use the z-based color
#     cmap="viridis",
#     point_size=5,           # adjust dot size
#     render_points_as_spheres=True,
# )
# plotter.set_background("white")
# plotter.show()