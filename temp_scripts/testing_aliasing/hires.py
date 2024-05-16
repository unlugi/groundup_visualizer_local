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

    img = Image.fromarray(depth)  # convert to PIL image for resize

    if resize:
        img = img.resize((img_size[0], img_size[0]), Image.BILINEAR, ) # Image.LANCZOS

    depth = np.array(img)

    return depth


def main():

    # High res
    path_m = 'hi-res/high_res_1024.exr'
    depth_m_raw = read_depth(path_m, resize=True, img_size=(256, 256))
    depth_m_raw = 1 - depth_m_raw
    depth_m = 5 * depth_m_raw

    cv2.imwrite('hi-res/high_res_256.exr', depth_m)

    # High res -> 512
    path_m_2 = 'hi-res/high_res_1024.exr'
    depth_m2_raw = read_depth(path_m_2, resize=True, img_size=(512, 512))
    depth_m2_raw = 1 - depth_m2_raw
    depth_m2 = 5 * depth_m2_raw

    cv2.imwrite('hi-res/high_res_512.exr', depth_m2)



    # low res
    path_l = 'hi-res/low_res_256.exr'
    depth_l_raw = read_depth(path_l, resize=False, img_size=(256, 256))
    depth_l_raw = 1 - depth_l_raw
    depth_l = 5 * depth_l_raw

    # low res
    path_l2 = 'hi-res/low_res_256.exr'
    depth_l2_raw = read_depth(path_l2, resize=True, img_size=(1024, 1024))
    depth_l2_raw = 1 - depth_l2_raw
    depth_l2 = 5 * depth_l2_raw

    cv2.imwrite('hi-res/low_res_1024.exr', depth_l2)

    # mesh both
    mesh_generator = BuildingMeshGenerator(use_color=False, mask_color=[], apply_dilation_mask=False)

    start_time = time.time()
    mesh_l = mesh_generator.generate_mesh(depths=depth_l,
                                          grid_size=(3.35, 3.35), )
    end_time = time.time()
    execution_time = end_time - start_time
    print('256x256')
    print(f"Execution time: {execution_time} seconds")

    start_time = time.time()
    mesh_l2 = mesh_generator.generate_mesh(depths=depth_l2,
                                          grid_size=(3.35, 3.35), )
    end_time = time.time()
    execution_time = end_time - start_time
    print('low_to_1024')
    print(f"Execution time: {execution_time} seconds")


    mesh_m = mesh_generator.generate_mesh(depths=depth_m,
                                          grid_size=(3.35, 3.35), )

    start_time = time.time()
    mesh_m2 = mesh_generator.generate_mesh(depths=depth_m2,
                                          grid_size=(3.35, 3.35), )
    end_time = time.time()
    execution_time = end_time - start_time
    print('512x512')
    print(f"Execution time: {execution_time} seconds")

    # save trimesh
    mesh_l.export("hi-res/low.obj", file_type='obj', include_color=True)
    mesh_l2.export("hi-res/low1024.obj", file_type='obj', include_color=True)
    mesh_m.export("hi-res/high.obj", file_type='obj', include_color=True)
    mesh_m2.export("hi-res/high512.obj", file_type='obj', include_color=True)

    # start_time = time.time()
    # ...
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"Execution time: {execution_time} seconds")


main()


