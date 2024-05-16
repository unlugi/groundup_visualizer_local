import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from PIL import Image

from utils.meshing_v2_utils import BuildingMeshGenerator

def read_depth(filepath):
    depth = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = depth[..., 2]
    # img = Image.fromarray(depth)  # convert to PIL image for resize
    return depth


# load .exr file

# zpass
path_z = 'zpass.exr'
depth_z = read_depth(path_z)
# depth_z = 5 - depth_z

# mist
path_m = 'mist_5.exr'
depth_m_raw = read_depth(path_m)
depth_m_raw = 1 - depth_m_raw

# normalize
# depth_m = (depth_m_raw - np.min(depth_m_raw)) / (np.max(depth_m_raw) - np.min(depth_m_raw))
# depth_m = 5 - (depth_m * 5)
# depth_m = depth_m_raw / (np.max(depth_m_raw) - np.min(depth_m_raw))
# depth_m = (1 - depth_m)
depth_m = 5 * depth_m_raw

# mesh both
mesh_generator = BuildingMeshGenerator(use_color=False, mask_color=[], apply_dilation_mask=False)


mesh_z = mesh_generator.generate_mesh(depths=depth_z,
                                         grid_size=(3.35, 3.35),)

mesh_m = mesh_generator.generate_mesh(depths=depth_m,
                                         grid_size=(3.35, 3.35),)

# save trimesh
mesh_z.export("zpass.obj", file_type='obj', include_color=True)
mesh_m.export("mist.obj", file_type='obj', include_color=True)














