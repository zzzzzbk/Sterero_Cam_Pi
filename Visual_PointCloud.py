import open3d as o3d
import tkinter as tk
from tkinter import filedialog
#import pyvista as pv
import numpy as np
import pyvista as pv
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

##Convert Open3D point cloud to PyVista mesh and visualize

# Extract points and colors from the point cloud
# points = np.asarray(pcd.points)
# colors = np.asarray(pcd.colors) if pcd.has_colors() else None

# # Create PyVista point cloud
# mesh = pv.PolyData(points)
# mesh["colors"] = (colors * 255).astype(np.uint8)

# # Visualize with PyVista
# mesh.plot(point_size=5, show_edges=False)