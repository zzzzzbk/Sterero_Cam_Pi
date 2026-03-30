import open3d as o3d
import tkinter as tk
from tkinter import filedialog
import argparse

import numpy as np
import os
import cv2
import imageio

# Parse command line arguments
parser = argparse.ArgumentParser(description="Rotating point cloud visualization")
parser.add_argument("--format", choices=["mp4", "gif"], default="mp4", 
                    help="Output format: mp4 or gif (default: mp4)")
args = parser.parse_args()

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



# Rotate 180 degrees around X-axis to correct upside-down orientation
rotation_matrix = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
pcd.rotate(rotation_matrix, center=np.array([0, 0, 0]))


# Visualize the point cloud
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Rotating Point Cloud", width=800, height=600)
vis.add_geometry(pcd)

# # Optional render tuning
# opt = vis.get_render_option()
# opt.point_size = 2.0

ctr = vis.get_view_control()

num_frames = 180
yaw_step = 15
fps = 30

# Determine output file format
file_ext = ".gif" if args.format == "gif" else ".mp4"
output_video = os.path.join(os.path.dirname(file_path), f"pointcloud_rotation{file_ext}")
os.makedirs(os.path.dirname(output_video), exist_ok=True)
#ctr.rotate(1000, 1280)  # Start with a view from the back

writer = None
frames_list = []  # For GIF saving

for _ in range(num_frames):
    ctr.rotate(yaw_step,0)
    vis.poll_events()
    vis.update_renderer()

    frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    frame = (frame * 255).astype(np.uint8)      # RGB uint8
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if args.format == "gif":
        # Store frames for GIF
        frames_list.append(frame)
    else:
        # MP4 saving
        if writer is None:
            h, w = frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        writer.write(frame_bgr)

vis.destroy_window()

# Save based on format
if args.format == "gif":
    imageio.mimsave(output_video, frames_list, fps=fps)
elif writer is not None:
    writer.release()

print(f"Saved: {output_video}")