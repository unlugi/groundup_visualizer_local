# from .visualizer_base import BaseVisualizer
from .visualizer_p3d import GroundUpVisualizerP3D
from utils.meshing_v2_utils import BuildingMeshGenerator
from utils.camera_utils import load_camera
from temp_scripts.test_mesh_and_camera_blender import RendererBlender

import os
import numpy as np
import torch
import bpy
import bmesh


class GroundUpVisualizerBlender(GroundUpVisualizerP3D):
    def __init__(self, sample_path, dataset_root, save_path, scene_name, samples_baseline, samples_sr, cfg_dict,
                 image_size=(256, 256), light_offset=(0, 0, 5),
                 add_color_to_mesh=None, device='cpu'):
        super().__init__(sample_path, dataset_root, save_path, scene_name, samples_baseline, samples_sr, image_size=image_size, light_offset=light_offset )
        self.device = self.get_device(device)
        self.cameras = self.parse_path_and_read_cameras()
        self.add_color_to_mesh = add_color_to_mesh
        self.mesh_generator = BuildingMeshGenerator(use_color=add_color_to_mesh, mask_color=self.masks,
                                                    apply_dilation_mask=True)

        self.renderer = RendererBlender(cfg=cfg_dict['cfg_blender'],
                                        cfg_vis=cfg_dict,
                                        mode='gt',
                                        cameras=self.cameras)

        self.mesh_dict = {'gt': None,
                          'pc': None,
                          'pred': None,
                          'pred_baseline': None,
                          }

        self.prepare_mesh_for_bpy(mode='pred')


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

    def convert_from_p3d_to_bpy(self, mesh, name='Building', collection_name='Collection'):
        vertices = mesh.verts_packed().detach().cpu().numpy()
        faces = mesh.faces_packed().detach().cpu().numpy()
        vertex_colors = mesh.textures._verts_features_padded.detach().cpu().numpy()[0]
        normals = mesh.faces_normals_padded().detach().cpu().numpy()

        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(mesh.name, mesh)
        col = bpy.data.collections[collection_name]
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        mesh.from_pydata(vertices, [], faces)

        # Update scene
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()

    def get_per_vertex_colors(self, mode='gt'):

        # Get the current mesh
        mesh_pytorch3d = self.mesh_dict[mode].clone()

        # Extract vertex colors
        vertex_colors_np = mesh_pytorch3d.textures._verts_features_padded.detach().cpu().numpy()
        vertex_colors_np = (vertex_colors_np[0] * 255.0).astype(np.uint8)

        # Append alpha channel for vertex colors
        vertex_colors_np = np.append(vertex_colors_np, np.ones((vertex_colors_np.shape[0], 1), dtype=np.uint8) * 255,
                                     axis=1)

        return vertex_colors_np


    def prepare_mesh_for_bpy(self, mode='gt'):

        # # 1) Create the mesh
        # self.get_mesh_in_world_coordinates('gt', update_face_colors=False)
        # # 2) Convert the trimesh to bpy mesh
        # self.convert_from_p3d_to_bpy(self.mesh_dict['gt'].clone())
        # # 3) Also get the vertex colors
        # ...
        # # 4) Put the new mesh in the scene
        # # 5) Render (for checking mesh works)
        #
        # # Create save path for current scene
        # save_path = os.path.join(self.save_path, self.sample_idx)
        # print(save_path)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # path_render = os.path.join(save_path, "render_{}_{}_blender.png".format('gt', self.sample_idx))
        # self.renderer.render(save_path=path_render)

        #1) Create the mesh
        self.get_mesh_in_world_coordinates(mode, update_face_colors=False)

        # 2) Export to file
        # Create save path for current scene
        save_path = os.path.join(self.save_path, self.sample_idx)
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.export_mesh_p3d(mesh_name='mesh_{}'.format(mode), mode=mode,
                             save_path=save_path, update_face_colors=False,
                             export_p3d=False)

        # 3) Import to Blender
        self.renderer.import_input_model(path_mesh=os.path.join(save_path, 'mesh_{}.obj'.format(mode)))


        # 3.5) enable vertex color material
        self.renderer.enable_material(mode=mode)

        # 4) Get vertex colors and assign to blender mesh (GET RID OF THIS)
        # vertex_colors = self.get_per_vertex_colors(mode=mode)
        # self.renderer.assign_vertex_colors_to_mesh(vertex_colors)

        # Create save path for current scene
        save_path = os.path.join(self.save_path, self.sample_idx)
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        path_render = os.path.join(save_path, "render_{}_{}_blender.png".format(mode, self.sample_idx))
        self.renderer.render(save_path=path_render)


