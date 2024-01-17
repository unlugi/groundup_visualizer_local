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
    swap_y_z = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, -1, 0, 0],
                         [0, 0, 0, 1]])

    rot_z = get_transformation_matrix('Z', degree=90)
    rot_y = get_transformation_matrix('Y', degree=0)
    rot_x = get_transformation_matrix('X', degree=0)

    c2w = swap_y_z @ blender_cam_matrix

    t = c2w[:3, -1]  # Extract translation of the camera
    r = c2w[:3, :3] @ rot_z @ rot_y @ rot_x # Extract rotation matrix of the camera

    t = t @ r  # Make rotation local

    return r , t ## OUTPUT IS MIRRORED WEIRDLY

def convert_camera_blender_2_pytorch3d_from_issue(blender_cam_matrix):
    """
    Convert camera matrix from Blender to Pytorch3D convention - used for rendering with Pytorch3D cameras
    :param blender_cam_matrix: 4x4 raw camera matrix from Blender
    :return: Camera pose r 3X3, t 3X1 in Pytorch3D convention
    """
    swap_y_z = np.array([[1, 0, 0, 0],
                         [0, 0, -1, 0],
                         [0, -1, 0, 0],
                         [0, 0, 0, 1]])

    flip_x = np.array([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]
                      )

    deg180 = np.deg2rad(180)
    rot_z = np.array([[np.cos(deg180), -np.sin(deg180), 0],
                      [np.sin(deg180), np.cos(deg180), 0],
                      [0, 0, 1]])

    c2w = swap_y_z @ blender_cam_matrix

    t = c2w[:3, -1]  # Extract translation of the camera
    r = c2w[:3, :3] @ rot_z @ flip_x  # Extract rotation matrix of the camera

    t = t @ r  # Make rotation local

    return r, t


def load_camera(camera_pose, cam_type='Camera', topdown_depth_offset=100.0):

    # Cast to float32
    camera_pose = camera_pose.astype(np.float32)

    # Convert to pytorch3D camera - from Blender to Pytorch3D
    camera_renderer_R, camera_renderer_T = convert_camera_blender_2_pytorch3d_from_issue(camera_pose)
    RT_ = np.concatenate((camera_renderer_R, camera_renderer_T[:, None]), axis=1)
    camera_renderer_RT = np.concatenate((RT_, np.array([[0, 0, 0, 1]])), axis=0)

    # Get Matrices for back-projecting depth maps to 3D point clouds (Blender2Pytorch3D)
    camera_point_cloud_R, camera_point_cloud_T = convert_camera_blender_2_pytorch3d_backproject_test(camera_pose)
    RT_pc = np.concatenate((camera_point_cloud_R.T, camera_point_cloud_T[:, None]), axis=1)
    camera_pc_RT = np.concatenate((RT_pc, np.array([[0, 0, 0, 1]])), axis=0)  # extrinsics

    if cam_type == 'Camera_Top_Down':
        camera_pc_RT[2, -1] = topdown_depth_offset

    return {'camera_renderer': camera_renderer_RT.astype(np.float32),
            'camera_pc': camera_pc_RT.astype(np.float32)}