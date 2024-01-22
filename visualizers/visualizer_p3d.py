import os
import time
import numpy as np
import torch
import PIL.Image as Image
import trimesh
from trimesh.exchange.obj import export_obj

from utils.mesh_metrics_utils import BackprojectDepth

from .visualizer_base import BaseVisualizer
from utils.camera_utils import load_camera
from utils.meshing_v2_utils import BuildingMeshGenerator, \
                                   update_vertex_colors, update_vertex_colors_fast, update_vertex_colors_fast_padding
from utils.p3d_utils import define_camera, mesh_renderer

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

class GroundUpVisualizerP3D(BaseVisualizer):
    def __init__(self, sample_path, dataset_root, scene_name, add_color_to_mesh=None, device='cpu', verbose=True):
        super().__init__(sample_path, dataset_root, scene_name)
        self.verbose = verbose
        
        self.device = self.get_device(device)
        self.masks = self.move_to_device(self.masks)
        self.cameras = self.parse_path_and_read_cameras()
        self.add_color_to_mesh = add_color_to_mesh
        self.mesh_generator = BuildingMeshGenerator(use_color=add_color_to_mesh, mask_color=self.masks, apply_dilation_mask=False)

        
    def get_device(self, device_name):
        # Set device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' if device_name=='cuda' else '-1'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print('Device:', device)
        return device

    def move_to_device(self, data):
        """
        Move data to device
        :param data: dict
        :return: each entry of data on device - hxw to 1xhxw
        """
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data[key] = torch.from_numpy(data[key])[None, ...].to(self.device)
            elif isinstance(data[key], dict):
                for key2 in data[key].keys():
                    if isinstance(data[key][key2], np.ndarray):
                        data[key][key2] = torch.from_numpy(data[key][key2]).to(self.device)
            else:
                raise NotImplementedError
        return data


    def fix_camera_intrinsics(self, K, image_size):
        # Modify K to match render resolution
        if image_size != self.orig_image_size:
            K[0] *= image_size[0]/ self.orig_image_size[0]
            K[1] *= image_size[0]/ self.orig_image_size[1]
        return K


    def parse_path_and_read_cameras(self):
        # Get paths for cam_td, cam_p, K_td, K_p
        path_cam_perspective = os.path.join(self.dataset_root, 'Camera', 'camera', 'campose_raw_{}.npz'.format(self.sample_idx))
        path_cam_topdown = os.path.join(self.dataset_root, 'Camera_Top_Down', 'camera', 'campose_raw_{}.npz'.format(self.sample_idx))
        path_K_perspective = os.path.join(self.dataset_root, 'cam_K.npy')
        path_K_topdown = os.path.join(self.dataset_root, 'cam_K_td.npy')

        # Get cameras and convert to Pytorch3D convention
        cam_perspective_raw = np.load(path_cam_perspective)['data']
        cam_topdown_raw = np.load(path_cam_topdown)['data']

        _, _, cam_T_world = self.from_blender_toWorld_to_OpenCV_worldTocam(blender_matrix_world=cam_perspective_raw, is_camera=True)
        cam_T_world = cam_T_world.astype(np.float32)

        cam_perspective = load_camera(cam_perspective_raw.copy(), cam_type='Camera')
        cam_topdown = load_camera(cam_topdown_raw.copy(), cam_type='Camera_Top_Down')

        # Get camera intrinsics K_td, invK_td, K_p, invK_p
        K_p = np.load(path_K_perspective).astype(np.float32)
        K_td = np.load(path_K_topdown).astype(np.float32)
        invK_p = np.linalg.inv(K_p)
        invK_td = np.linalg.inv(K_td)

        return {'cam_T_world': cam_T_world, # opencv 
                'cam_perspective': cam_perspective, 'cam_topdown': cam_topdown, # Pytorch3D
                'K_p': K_p, 'K_td': K_td,
                'invK_p': invK_p, 'invK_td': invK_td}


    def from_blender_toWorld_to_OpenCV_worldTocam(self, blender_matrix_world, is_camera=False):
        # 1. To satisfy both camera and other objects, the transformation matrix from Blender to OpenCV is the following:
        R_BlenderView_to_OpenCVView = np.diag([1 if is_camera else -1, -1, -1])

        # 2. From Blender get the world transformation matrix (to_world - Blender Global) by doing obj.matrix_world() then decompose to location and rotation
        location = blender_matrix_world[:3, -1]
        rotation = blender_matrix_world[:3, :3]
        
        trans = np.eye(4)
        trans[:3, :3] = R_BlenderView_to_OpenCVView
        
        world_T_cam = blender_matrix_world @ trans
        
        cam_T_world = np.linalg.inv(world_T_cam)
        
        return None, None, cam_T_world

    def find_camera_bounds_world_3d(self):

        # Get the 3D world coordinates of the ground plane
        corners_ground_uv1 = np.array([ [  0, 0,   1], # tl
                                        [256, 0,   1], # tr
                                        [  0, 256, 1], # bl
                                        [256, 256, 1], # br
                                 ])

        # Max depth of the ground plane - offset for orthographic2perspective projection
        max_depth = self.data['gt'].max()
        max_depth = max_depth + (100.0 - 5.0) # TODO: hard-coded

        # homogenous coordinates = [z*x z*y z*1 1]
        corners_ground_uv1 = np.concatenate( [max_depth * corners_ground_uv1, np.ones_like(corners_ground_uv1[:, 0][:, None])], axis=1 )
        corners_ground_uv1 = torch.from_numpy(corners_ground_uv1.astype(np.float32)).to(self.device)

        # Get camera intrinsics and extrinsics
        Kinv_td = torch.from_numpy(self.cameras['invK_td']).to(self.device)
        extrinsics_RT_td = torch.from_numpy(self.cameras['cam_topdown']['camera_pc']).to(self.device)

        # Back-project to 3D world coordinates
        coords_cam = Kinv_td @ corners_ground_uv1.T
        corners_ground_xyz = extrinsics_RT_td.inverse() @ coords_cam
        corners_ground_xyz = corners_ground_xyz.type(torch.float32) # xzy?
        corners_ground_xyz = corners_ground_xyz[:3, :].T # 4X3

        return corners_ground_xyz
    
    def find_mesh_world_coordinates_3d(self, mode='gt'):

        if self.verbose:
            print('Generating mesh...')
        
        threshold = 10000
        depth_pixels_values = self.data[mode][self.data[mode] < threshold][:,None]

        # offset if ground too low
        max_depth = self.data[mode].max()
        offset_ground = self.data['gt'].max() - self.data[mode].max()

        depth_pixels_values = depth_pixels_values + offset_ground + (100.0 - 5.0) # TODO: hard-coded

        depth_pixels_xy = np.argwhere(self.data[mode] < threshold)

        # Get foreground pixels xy and values depth_pixels_2d_homogeneous

        # depth_pixels_2d_homogeneous = [z*x z*y z*1 1]
        vertices_uv1 = np.concatenate( [depth_pixels_values * depth_pixels_xy, 
                                        depth_pixels_values, 
                                        np.ones_like(depth_pixels_values)], axis=1).astype(np.float32)
        vertices_uv1 = torch.from_numpy(vertices_uv1.astype(np.float32)).to(self.device)

        # Get camera intrinsics and extrinsics
        Kinv_td = torch.from_numpy(self.cameras['invK_td']).to(self.device)
        pose = torch.from_numpy(self.cameras['cam_topdown']['camera_pc']).to(self.device)

        # Back-project to 3D world coordinates
        vertices_cam = Kinv_td @ vertices_uv1.T
        # vertices_xyz = extrinsics_RT_td.inverse() @ vertices_cam
        vertices_xyz = pose @ vertices_cam
        vertices_xyz = vertices_xyz.type(torch.float32) # xzy?
        vertices_xyz = vertices_xyz[:3, :].T # 4X3
        
        # depth_b1hw = torch.tensor(self.data[mode][None, None]).to(self.device)
        # depth_b1hw += offset_ground + (100.0 - 5.0) 
        # backprojector = BackprojectDepth(height=256, width=256)
        # backprojector = backprojector.to(self.device)
        # points_14N = backprojector(depth_b1hw, Kinv_td.unsqueeze(0))
        # points_14N = pose.unsqueeze(0) @ points_14N
        # vertices_xyz = points_14N.squeeze(0)[:3, :].T

        return vertices_xyz

    def run_meshing(self, model_name='gt', cam_bounds=None):

        # Mesh the gt depth map - trimesh object
        if model_name == 'gt':
            mesh = self.mesh_generator.generate_mesh(depths=self.data['gt'],
                                                     grid_size=(3.35, 3.35),)
        elif model_name == 'pred':
            mesh = self.mesh_generator.generate_mesh(depths=self.data['pred'],
                                                     grid_size=(3.35, 3.35),
                                                     )
        else: # TODO pointcloud proj
            raise NotImplementedError
        return mesh

    def get_mesh_in_world_coordinates(self, model_name='gt', update_face_colors=False):
        # Find the 3D world coordinates of the ground plane
        cam_bounds_xyz = self.find_camera_bounds_world_3d()

        # # Align the mesh vertices to 3D world coordinate bounds of the scene.
        mesh_elevation = self.run_meshing(model_name=model_name, cam_bounds=cam_bounds_xyz)

        # verts = torch.tensor(mesh_elevation.vertices.copy(), dtype=torch.float32).to(self.device)
        faces = torch.tensor(mesh_elevation.faces.copy(), dtype=torch.int64).to(self.device)
        normals = torch.tensor(mesh_elevation.vertex_normals.copy(), dtype=torch.float32).to(self.device)

        # Initialize each vertex to be white
        # verts_rgb = 0.6 * torch.ones_like(verts)[None]
        vertex_colors = torch.tensor((mesh_elevation.visual.vertex_colors/255.0)[:,:3][None], dtype=torch.float32).to(self.device)
        textures = Textures(verts_rgb=vertex_colors)

        # Transform this to 3D world coordinate space this time
        verts_transformed = self.find_mesh_world_coordinates_3d(mode=model_name)

        # Create a PyTorch3D Meshes object
        self.mesh = Meshes(verts=[verts_transformed], faces=[faces], verts_normals=[normals], textures=textures)
        # self.mesh = Meshes(verts=[verts_transformed], faces=[faces], textures=textures)

        if update_face_colors:
            target_color_value = np.array([98, 227, 132]) / 255.0

            start = time.time()
            # Update vertex colors based on the target color
            # self.mesh = update_vertex_colors(self.mesh, target_color_value)
            self.mesh = update_vertex_colors_fast(self.mesh, target_color_value)
            # self.mesh = update_vertex_colors_fast_padding(self.mesh, target_color_value)
            end = time.time()
            print('time it took:{} seconds '.format(end-start))


    def render_scene(self, image_size=(256, 256), offset=(0, 0, 0)):

        if self.verbose:
            print('Rendering scene...')

        # Create pytorch3D cameras
        K = torch.from_numpy(self.cameras['K_p']).to(self.device).clone()
        R_for_camera_perspective = self.cameras['cam_perspective']['camera_renderer'][:3, :3]
        t_for_camera_perspective = self.cameras['cam_perspective']['camera_renderer'][:3, -1]


        # Fix camera K
        K = self.fix_camera_intrinsics(K, image_size)

        # Define pytorch3D camera
        cameras_perspective = define_camera(K[None, :, :],
                                            image_size,
                                            R_for_camera_perspective[None, ...],
                                            t_for_camera_perspective[None, ...],
                                            device=self.device)

        # Render the scene
        renderer_ = mesh_renderer( cameras=cameras_perspective, imsize=image_size, device=self.device, offset=offset)
        perspective_render = renderer_(self.mesh)

        perspective_color = perspective_render[0, :].detach().cpu().numpy()
        perspective_color = Image.fromarray((perspective_color * 255.0).astype(np.uint8))
        perspective_color = perspective_color.convert('RGBA')

        background_color = (255, 255, 255, 255)  # RGBA value for white
        image_w_bg = Image.new("RGBA", perspective_color.size, background_color)

        # Copy RGB values from the original image and set alpha to 255 (fully opaque)
        image_w_bg.paste(perspective_color, (0, 0), mask=perspective_color)

        return cameras_perspective, image_w_bg


    def export_mesh_p3d(self, mesh_name, save_path, update_face_colors=True):

        mesh_pytorch3d = self.mesh

        if update_face_colors:
            target_color_value = np.array([98, 227, 132])/ 255.0

            # Update vertex colors based on the target color
            mesh_pytorch3d = update_vertex_colors(mesh_pytorch3d, target_color_value)

        # Convert PyTorch3D mesh to trimesh
        verts_np = mesh_pytorch3d.verts_packed().detach().cpu().numpy()
        faces_np = mesh_pytorch3d.faces_packed().detach().cpu().numpy()
        vertex_colors_np = mesh_pytorch3d.textures._verts_features_padded.detach().cpu().numpy()
        vertex_colors_np =(vertex_colors_np * 255.0).astype(np.uint8)

        mesh_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=vertex_colors_np)
        mesh_trimesh.visual.vertex_colors = vertex_colors_np
        trimesh.repair.fix_normals(mesh_trimesh, multibody=False)

        # filename = os.path.join(save_path, "mesh_{}_{}.obj".format(mesh_name, self.sample_idx))
        filename = os.path.join(save_path, mesh_name)
        mesh_trimesh.export(filename+".obj", file_type='obj', include_color=True)
        # mesh_with_color = export_obj(mesh_trimesh, include_color=True, include_normals=True)
        # mesh_with_color.export(filename, file_type='obj', include_color=True)

    def get_trimesh(self, update_face_colors=True):
        
        mesh_pytorch3d = self.mesh


        if update_face_colors:
            target_color_value = np.array([98, 227, 132])/ 255.0

            # Update vertex colors based on the target color
            mesh_pytorch3d = update_vertex_colors(mesh_pytorch3d, target_color_value)

        # Convert PyTorch3D mesh to trimesh
        verts_np = mesh_pytorch3d.verts_packed().detach().cpu().numpy()
        faces_np = mesh_pytorch3d.faces_packed().detach().cpu().numpy()
        vertex_colors_np = mesh_pytorch3d.textures._verts_features_padded.detach().cpu().numpy()
        vertex_colors_np =(vertex_colors_np * 255.0).astype(np.uint8)

        mesh_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=vertex_colors_np)
        mesh_trimesh.visual.vertex_colors = vertex_colors_np
        
        trimesh.repair.fix_normals(mesh_trimesh, multibody=False)
        
        return mesh_trimesh



