import numpy as np

def get_transformation_matrix(axis, degree):

    deg = np.deg2rad(degree)
    if axis == 'X':
        rot = np.array([[1, 0, 0],
                          [0, np.cos(deg), -np.sin(deg)],
                          [0, np.sin(deg), np.cos(deg)]
                          ])

    elif axis == 'Y':
        rot = np.array([[np.cos(deg), 0, np.sin(deg)],
                          [0, 1, 0],
                          [-np.sin(deg), 0, np.cos(deg)]
                          ])

    elif axis == 'Z':
        rot = np.array([[np.cos(deg), -np.sin(deg), 0],
                          [np.sin(deg), np.cos(deg), 0],
                          [0, 0, 1]])
    else:
        rot = np.eye(3)

    return rot

def convert_camera_blender_2_pytorch3d_backproject_test(blender_cam_matrix):
    """
    Convert camera matrix from Blender to Pytorch3D convention - used for back-projecting depth maps to 3D point clouds
    :param blender_cam_matrix: blender_cam_matrix: 4x4 raw camera matrix from Blender
    :return: Camera extrinsics r 3X3, t 3X1 in Pytorch3D convention
    """

    mirror_axes = np.array([[-1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, -1, 0],
                         [0, 0, 0, 1]])

    c2w = blender_cam_matrix @ mirror_axes

    return c2w

def convert_from_pytorch3d_to_opencv(pytorch3d_cam_matrix):
    """
    Convert camera matrix from Blender to Pytorch3D convention - used for back-projecting depth maps to 3D point clouds
    :param blender_cam_matrix: blender_cam_matrix: 4x4 raw camera matrix from Blender
    :return: Camera extrinsics r 3X3, t 3X1 in Pytorch3D convention
    """

    mirror_axes = np.array([[-1, 0, 0, 0],
                         [0, -1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    c2w = pytorch3d_cam_matrix @ mirror_axes

    return c2w

# def convert_camera_blender_2_pytorch3d_from_issue(blender_cam_matrix):
#     """
#     Convert camera matrix from Blender to Pytorch3D convention - used for rendering with Pytorch3D cameras
#     :param blender_cam_matrix: 4x4 raw camera matrix from Blender
#     :return: Camera pose r 3X3, t 3X1 in Pytorch3D convention
#     """

#     mirror_axes = np.array([[-1, 0, 0, 0],
#                          [0, 1, 0, 0],
#                          [0, 0, -1, 0],
#                          [0, 0, 0, 1]])

#     c2w =  blender_cam_matrix @ mirror_axes

#     t = c2w[:3, -1]
#     r = c2w[:3, :3]

#     return r, t


def load_camera(camera_pose, cam_type='Camera', topdown_depth_offset=100.0):

    # Cast to float32
    camera_pose = camera_pose.astype(np.float32).copy()
    
    if cam_type == 'Camera_Top_Down':
        camera_pose[2, -1] = topdown_depth_offset

    camera_pc = convert_camera_blender_2_pytorch3d_backproject_test(camera_pose)
    camera_renderer = np.linalg.inv(camera_pc)

    return {'camera_renderer': camera_renderer.astype(np.float32),
            'camera_pc': camera_pc.astype(np.float32)}