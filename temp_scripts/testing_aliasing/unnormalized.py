import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from PIL import Image
import time

from utils.meshing_v2_utils import BuildingMeshGenerator

def read_depth(filepath, resize=False, img_size=(256, 256)):
    depth = cv2.imread(filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = depth[..., 2]

    if resize:
        img = Image.fromarray(depth)  # convert to PIL image for resize
        img = img.resize((img_size[0], img_size[0]), Image.BILINEAR, ) # Image.LANCZOS
        depth = np.array(img)

    return depth


def main():

    idx = '0'
    sample_idx = "0000"

    # zpass
    path_z = 'data_final_version/depth_{}.exr0001.exr'.format(sample_idx)
    depth_z_raw = read_depth(path_z, resize=False, img_size=(256, 256))
    depth_z = depth_z_raw

    # mist
    path_m = 'data_final_version/mist_{}.exr0001.exr'.format(sample_idx)
    depth_m_raw = read_depth(path_m, resize=False, img_size=(256, 256))

    # # normalize mist pass
    depth_m_normalized = (depth_m_raw - depth_m_raw.min()) / (depth_m_raw.max() - depth_m_raw.min())
    depth_m = depth_z.min() + (depth_z.max() - depth_z.min()) * depth_m_normalized

    # mesh both
    mesh_generator = BuildingMeshGenerator(use_color=False, mask_color=[], apply_dilation_mask=False)

    start_time = time.time()
    mesh_z = mesh_generator.generate_mesh(depths=depth_z,
                                          grid_size=(0.32, 0.32), )
    end_time = time.time()
    execution_time = end_time - start_time
    print('zpass')
    print(f"Execution time: {execution_time} seconds")

    start_time = time.time()
    mesh_m = mesh_generator.generate_mesh(depths=depth_m,
                                          grid_size=(0.32, 0.32), )
    end_time = time.time()
    execution_time = end_time - start_time
    print('mist')
    print(f"Execution time: {execution_time} seconds")


    # save trimesh
    mesh_z.export("data_final_version/depth_{}.obj".format(idx), file_type='obj', include_color=True)
    mesh_m.export("data_final_version/mist_{}.obj".format(idx), file_type='obj', include_color=True)


main()


