import os
import cairosvg
import cv2
import numpy as np
import xml.etree.ElementTree as ET

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def load_vector_image(svg_path):
    ...


def load_depth_map(path_depth):
    depth_map = cv2.imread(path_depth, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_map = depth_map[..., 2]
    return depth_map



# SVG and depth map file paths
data_folder = "data/"
svg_file_path = os.path.join(data_folder, "sketch/0002.svg")
depth_map_file_path = os.path.join(data_folder, "depth/Image0002.exr")

# Load depth map
depth_map = load_depth_map(depth_map_file_path)



