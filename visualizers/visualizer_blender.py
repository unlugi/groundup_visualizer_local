# from .visualizer_base import BaseVisualizer
from .visualizer_p3d import GroundUpVisualizerP3D
from utils.meshing_v2_utils import BuildingMeshGenerator
from utils.camera_utils import load_camera

import os
import numpy as np
import torch
import bpy

class GroundUpVisualizerBlender(GroundUpVisualizerP3D):
    def __init__(self, sample_path, dataset_root, scene_name, add_color_to_mesh=None, device='cpu'):
        super().__init__(sample_path, dataset_root, scene_name, add_color_to_mesh, device)

        # Get cameras - keep the raw camera matrices for Blender
        self.cameras = self.parse_path_and_read_cameras()

        # Mesh generator
        self.add_color_to_mesh = add_color_to_mesh
        self.mesh_generator = BuildingMeshGenerator(use_color=add_color_to_mesh,
                                                    mask_color=self.masks,
                                                    apply_dilation_mask=False)

    def parse_path_and_read_cameras(self):
        # Get paths for cam_td, cam_p, K_td, K_p
        path_cam_perspective = os.path.join(self.dataset_root, 'Camera', 'camera', 'campose_raw_{}.npz'.format(self.sample_idx))
        path_cam_topdown = os.path.join(self.dataset_root, 'Camera_Top_Down', 'camera', 'campose_raw_{}.npz'.format(self.sample_idx))
        path_K_perspective = os.path.join(self.dataset_root, 'cam_K.npy')
        path_K_topdown = os.path.join(self.dataset_root, 'cam_K_td.npy')

        # Get cameras and convert to Pytorch3D convention
        cam_perspective_raw = np.load(path_cam_perspective)['data']
        cam_topdown_raw = np.load(path_cam_topdown)['data']

        cam_perspective = load_camera(cam_perspective_raw.copy(), cam_type='Camera')
        cam_topdown = load_camera(cam_topdown_raw.copy(), cam_type='Camera_Top_Down')

        # Get camera intrinsics K_td, invK_td, K_p, invK_p
        K_p = np.load(path_K_perspective).astype(np.float32)
        K_td = np.load(path_K_topdown).astype(np.float32)
        invK_p = np.linalg.inv(K_p)
        invK_td = np.linalg.inv(K_td)

        return {'cam_perspective_raw': cam_perspective_raw, 'cam_topdown_raw': cam_topdown_raw, # Blender
                'cam_perspective': cam_perspective, 'cam_topdown': cam_topdown, # Pytorch3D
                'K_p': K_p, 'K_td': K_td,
                'invK_p': invK_p, 'invK_td': invK_td}


    def convert_to_blender_mesh(self):
        # Extract vertices and faces from PyTorch3D mesh
        vertices = self.mesh.verts_packed().detach().cpu().numpy()
        faces = self.mesh.faces_packed().detach().cpu().numpy()

        # Create a new mesh
        mesh = bpy.data.meshes.new(name="CustomMesh")
        obj = bpy.data.objects.new("CustomObject", mesh)
        bpy.context.collection.objects.link(obj)

        # Transfer PyTorch3D mesh data to Blender mesh
        mesh.from_pydata(vertices, [], faces)

        # Update the mesh
        mesh.update()

        print('here!')




    #
    # def render_scene(self):
    #     ...
    #
    # def export_mesh(self):
    #     ...




