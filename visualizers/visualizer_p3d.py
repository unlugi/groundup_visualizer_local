import os
import time
import numpy as np
import torch
import PIL.Image as Image
import trimesh
from trimesh.exchange.obj import export_obj

from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO

from utils.mesh_metrics_utils import BackprojectDepth

from .visualizer_base import BaseVisualizer
from utils.camera_utils import load_camera
from utils.meshing_v2_utils import BuildingMeshGenerator, get_xy_depth_homogeneous_coordinates_bs1_vis, \
                                   update_vertex_colors, update_vertex_colors_fast, update_vertex_colors_fast_padding
from utils.p3d_utils import define_camera, mesh_renderer

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from .pyrenderer import Renderer as pyRenderer, camera_marker, transform_trimesh
from .pyrenderer import create_light_array  
import pyrender

import matplotlib.pyplot as plt


class GroundUpVisualizerP3D(BaseVisualizer):
    def __init__(self, sample_path, dataset_root, scene_name, add_color_to_mesh=None, device='cpu', verbose=True):
        super().__init__(sample_path, dataset_root, scene_name)
        self.verbose = verbose
        
        self.device = self.get_device(device)
        self.masks = self.move_to_device(self.masks)
        self.cameras = self.parse_path_and_read_cameras()
        self.add_color_to_mesh = add_color_to_mesh
        self.mesh_generator = BuildingMeshGenerator(use_color=add_color_to_mesh, mask_color=self.masks, apply_dilation_mask=True)
        
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
        path_cam_perspective = os.path.join(self.dataset_root, 'Camera', 'camera','campose_raw_{}.npz'.format(self.sample_idx))
        path_cam_topdown = os.path.join(self.dataset_root, 'Camera_Top_Down', 'camera', 'campose_raw_{}.npz'.format(self.sample_idx))
        path_K_perspective = os.path.join(self.dataset_root, 'cam_K.npy')
        path_K_topdown = os.path.join(self.dataset_root, 'cam_K_td.npy')

        # Get cameras and convert to Pytorch3D convention
        cam_perspective = np.load(path_cam_perspective)['data']
        cam_topdown = np.load(path_cam_topdown)['data']

        cam_perspective = load_camera(cam_perspective, cam_type='Camera')
        cam_topdown = load_camera(cam_topdown, cam_type='Camera_Top_Down')

        # Get camera intrinsics K_td, invK_td, K_p, invK_p
        K_p = np.load(path_K_perspective).astype(np.float32)
        K_td = np.load(path_K_topdown).astype(np.float32)
        invK_p = np.linalg.inv(K_p)
        invK_td = np.linalg.inv(K_td)

        return {'cam_perspective': cam_perspective, 'cam_topdown': cam_topdown,
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
        depth_pixels_values = self.data[mode][self.data[mode] < threshold][:,None].copy()

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
        extrinsics_RT_td = torch.from_numpy(self.cameras['cam_topdown']['camera_pc']).to(self.device)

        # Back-project to 3D world coordinates
        vertices_cam = Kinv_td @ vertices_uv1.T
        vertices_xyz = extrinsics_RT_td.inverse() @ vertices_cam
        vertices_xyz = vertices_xyz.type(torch.float32) # xzy?
        vertices_xyz = vertices_xyz[:3, :].T # 4X3

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

    def transform_mesh_to_world_coordinates(self, p3d_mesh):
        extrinsics_RT_td = torch.from_numpy(self.cameras['cam_topdown']['camera_pc']).to(self.device)
        # vertices_xyz = extrinsics_RT_td.inverse() @ vertices_cam
        

    def get_mesh_in_world_coordinates(self, model_name='gt', update_face_colors=False):
        # Find the 3D world coordinates of the ground plane
        cam_bounds_xyz = self.find_camera_bounds_world_3d()

        # # Align the mesh vertices to 3D world coordinate bounds of the scene.
        mesh_elevation = self.run_meshing(model_name=model_name, cam_bounds=cam_bounds_xyz)

        # verts = torch.tensor(mesh_elevation.vertices.copy(), dtype=torch.float32).to(self.device)
        faces = torch.tensor(mesh_elevation.faces.copy(), dtype=torch.int64).to(self.device)
        faces = torch.flip(faces, [1])
        normals = torch.tensor(mesh_elevation.vertex_normals.copy(), dtype=torch.float32).to(self.device)
        # normals[:,0] = -normals[:,0]
        # normals[:,1] = -normals[:,1]

        # Initialize each vertex to be white
        # verts_rgb = 0.6 * torch.ones_like(verts)[None]
        vertex_colors = torch.tensor((mesh_elevation.visual.vertex_colors/255.0)[:,:3][None], dtype=torch.float32).to(self.device)
        textures = Textures(verts_rgb=vertex_colors)

        # Transform this to 3D world coordinate space this time
        verts_transformed = self.find_mesh_world_coordinates_3d(mode=model_name)
        # verts_transformed = torch.tensor(mesh_elevation.vertices).cuda()

        # Create a PyTorch3D Meshes object
        self.mesh = Meshes(verts=[verts_transformed], faces=[faces], verts_normals=None, textures=textures)
        # self.mesh_dict[model_name] = Meshes(verts=[verts_transformed], faces=[faces], verts_normals=[normals], textures=textures)

        if update_face_colors:
            print("Fixing mesh colors...")
            # target_color_value = np.array([98, 227, 132]) / 255.0 #
            target_color_value = np.array([250,250,250]) / 255.0

            start = time.time()
            # Update vertex colors based on the target color
            self.mesh = update_vertex_colors_fast(self.mesh, target_color_value)
            # self.mesh = update_vertex_colors_fast_padding(self.mesh, target_color_value)
            end = time.time()
            print('time it took:{} seconds '.format(end-start))


    def render_scene(self, image_size=(256, 256), offset=(0, 0, 0)):

        if self.verbose:
            print('Rendering scene...')

        # Create pytorch3D cameras
        K = torch.from_numpy(self.cameras['K_p']).to(self.device).clone()
        R_for_camera_perspective = self.cameras['cam_perspective']['camera_renderer'][:3, :3].copy()
        t_for_camera_perspective = self.cameras['cam_perspective']['camera_renderer'][:3, -1].copy()


        # Fix camera K
        K = self.fix_camera_intrinsics(K, image_size)

        # Define pytorch3D camera
        cameras_perspective = define_camera(K[None, :, :],
                                            image_size,
                                            R_for_camera_perspective[None, ...],
                                            t_for_camera_perspective[None, ...],
                                            device=self.device)

        # Render the scene
        renderer_ = mesh_renderer(cameras=cameras_perspective, imsize=image_size, device=self.device, offset=offset)
        perspective_render = renderer_(self.mesh)

        perspective_color = perspective_render[0, :].detach().cpu().numpy()
        perspective_color = Image.fromarray((perspective_color * 255.0).astype(np.uint8))
        perspective_color = perspective_color.convert('RGBA')

        background_color = (255, 255, 255, 255)  # RGBA value for white
        image_w_bg = Image.new("RGBA", perspective_color.size, background_color)

        # Copy RGB values from the original image and set alpha to 255 (fully opaque)
        image_w_bg.paste(perspective_color, (0, 0), mask=perspective_color)

        return cameras_perspective, image_w_bg

    def render_scene_pyrender(self, image_size=(256, 256), offset=(0, 0, 0), topdown=False):

        if self.verbose:
            print('Rendering scene...')

        if topdown:
            K = torch.from_numpy(self.cameras['K_td']).to(self.device).clone()
            pytorch3d_pose = np.linalg.inv(self.cameras['cam_topdown']['camera_pc'].copy())
        else:
            K = torch.from_numpy(self.cameras['K_p']).to(self.device).clone()
            pytorch3d_pose = np.linalg.inv(self.cameras['cam_perspective']['camera_pc'].copy())
        
        # Fix camera K
        K = self.fix_camera_intrinsics(K, image_size)
        
        R_BlenderView_to_OpenCVView = np.diag([-1, -1, 1, 1])
        rot_transform = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0,0,1,0],
            [0,0,0,1],
        ])
        opencv_pose = pytorch3d_pose @ rot_transform @ R_BlenderView_to_OpenCVView 

        trimesh_mesh = self.get_trimesh(update_face_colors=True)
        

        # light_pos = opencv_pose.copy()
        light_pos = opencv_pose.copy()
        vertices_center = trimesh_mesh.vertices.mean(axis=0)
        # light_pos[:3, 3] = vertices_center
        # light_pos[2, 3] -= 5.0
        light_pos = np.eye(4)
        light_pos[:3, 3] = np.array(vertices_center)
        light_pos[1, 3] += 2

        lights = create_light_array(
            pyrender.PointLight(intensity=4), 
            light_pos, 
            x_length=1,
            y_length=1,
            num_x=2,
            num_y=2,
        )

        meshes = [trimesh_mesh]
        mesh_materials = [None]
        
        # sphere = trimesh.creation.icosphere(radius=0.1)
        # sphere = transform_trimesh(sphere, light_pos)
        # meshes.append(sphere)
        # mesh_materials.append(pyrender.MetallicRoughnessMaterial(metallicFactor=0.0))
        

        pyrenderer = pyRenderer(image_size[0], image_size[1])
        render = pyrenderer.render_mesh(
            meshes, 
            image_size[0], image_size[1], 
            opencv_pose,
            K, True, mesh_materials=mesh_materials, 
            lights=lights,
            render_flags=pyrender.RenderFlags.SKIP_CULL_FACES
        )
        


        # render = np.transpose(render, axes=[1,0,2])
        image_w_bg = Image.fromarray(render)
        image_w_bg = image_w_bg.rotate(180)
        # image_w_bg.save("test.png")

        return None, image_w_bg

    def get_camera_marker_mesh(self, image_size = [256, 256], topdown=False):
        if topdown:
            K = torch.from_numpy(self.cameras['K_td']).to(self.device).clone()
            pytorch3d_pose = np.linalg.inv(self.cameras['cam_topdown']['camera_pc'].copy())
        else:
            K = torch.from_numpy(self.cameras['K_p']).to(self.device).clone()
            pytorch3d_pose = np.linalg.inv(self.cameras['cam_perspective']['camera_pc'].copy())

        # Fix camera K
        K = self.fix_camera_intrinsics(K, image_size)

        R_BlenderView_to_OpenCVView = np.diag([-1, -1, 1, 1])
        rot_transform = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0,0,1,0],
            [0,0,0,1],
        ])
        opencv_pose = pytorch3d_pose @ rot_transform @ R_BlenderView_to_OpenCVView 

        K = K.detach().cpu().numpy()
        fpv_camera = trimesh.scene.Camera(
            resolution=(image_size[0], image_size[1]), 
            focal=(K[0][0], K[1][1])
        )
        cam_marker_mesh = camera_marker(fpv_camera, cam_marker_size=0.3, sphere_rad=0.06)[1]
        
        cam_mesh = transform_trimesh(cam_marker_mesh, opencv_pose)
        
        return cam_mesh



    def get_trimesh(self, update_face_colors=True):
        
        mesh_pytorch3d = self.mesh

        if update_face_colors:
            target_color_value = np.array([98, 227, 132])/ 255.0

            # Update vertex colors based on the target color
            mesh_pytorch3d = update_vertex_colors_fast(mesh_pytorch3d, target_color_value)

        # Convert PyTorch3D mesh to trimesh
        verts_np = mesh_pytorch3d.verts_packed().detach().cpu().numpy()
        faces_np = mesh_pytorch3d.faces_packed().detach().cpu().numpy()
        vertex_colors_np = mesh_pytorch3d.textures._verts_features_padded.detach().cpu().numpy()
        vertex_colors_np =(vertex_colors_np * 255.0).astype(np.uint8)[0]

        mesh_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=vertex_colors_np)
        mesh_trimesh.visual.vertex_colors = vertex_colors_np
        
        # trimesh.repair.fix_normals(mesh_trimesh, multibody=False)
        
        return mesh_trimesh


        
    def get_pointcloud_depth_perspective_in_world_coordinates(self, mode='gt', is_save=False,
                                                              path_to_save=None, minmax_valid_depth=(0.01, 4.0)):

        depth_map_perspective = self.data['gt_perspective'].copy()
        depth_map_perspective = torch.from_numpy(depth_map_perspective).to(self.device).unsqueeze(0)

        # Get the float valid mask
        mask_for_valid_depth = ((depth_map_perspective >= minmax_valid_depth[0])
                                & (depth_map_perspective < minmax_valid_depth[1]))
        depth_map_perspective[~mask_for_valid_depth] = torch.tensor(np.nan)


        # UV coordinates of the depth map - [uv1]
        depth_pixels_2d_homogeneous, foreground_mask = get_xy_depth_homogeneous_coordinates_bs1_vis(depth_map_perspective,
                                                                                      mask_for_valid_depth,
                                                                                      )

        # Get 2D homogeneous coordinates from depth pixels
        # coords_homogeneous_uv1 = depth_pixels_2d_homogeneous.unsqueeze(0)
        coords_homogeneous_uv1 = depth_pixels_2d_homogeneous

        # Get camera intrinsics and extrinsics
        Kinv = self.cameras['invK_p'].copy()
        Kinv = torch.from_numpy(Kinv).type(torch.float32).to(self.device)
        extrinsics_RT_td = self.cameras['cam_perspective']['camera_pc'].copy()
        extrinsics_RT_td = torch.from_numpy(extrinsics_RT_td).type(torch.float32).to(self.device)

        # Backproject to 3d
        coords_cam = Kinv @ coords_homogeneous_uv1.T
        pts3D_topdown = extrinsics_RT_td.inverse() @ coords_cam

        # pts3D_topdown = pts3D_topdown.type(torch.float32)

        # pts3D_topdown_ = pts3D_topdown[:3, :].view(3, 256, 256)
        # pts3D_topdown_ = pts3D_topdown_[:, foreground_mask]

        point_cloud_td = Pointclouds(points=pts3D_topdown[:3, :].T[None, ...])

        if is_save:
            # filename_pc = os.path.join(path_to_save.format(mode, self.sample_idx))/
            IO().save_pointcloud(point_cloud_td, path_to_save, colors_as_uint8=False)

        return point_cloud_td
    
    def get_topdown_depth_map_viz(self):
        
        topdown_gt = self.data['gt'].copy()
        topdown_gt = torch.tensor(topdown_gt)[None]
        topdown_gt_torch, vmin, vmax = colormap_image(topdown_gt, return_vminvmax=True)
        
        topdown_gt_pil = Image.fromarray(np.uint8(255*topdown_gt_torch.detach().permute(1,2,0).cpu().numpy()))
        
        
        topdown_pred = self.data['pred'].copy()
        topdown_pred = torch.tensor(topdown_pred)[None]
        topdown_pred_torch = colormap_image(topdown_pred, vmin=vmin, vmax=vmax)
        topdown_pred_pil = Image.fromarray(np.uint8(255*topdown_pred_torch.detach().permute(1,2,0).cpu().numpy()))
        
        
        return topdown_gt_pil, topdown_pred_pil

    def get_pers_depth_map_viz(self):
        
        # topdown_gt = self.data['gt'].copy()
        # topdown_gt = torch.tensor(topdown_gt)[None]
        # topdown_gt_torch, vmin, vmax = colormap_image(topdown_gt, return_vminvmax=True)
        
        # topdown_gt_pil = Image.fromarray(np.uint8(255*topdown_gt_torch.detach().permute(1,2,0).cpu().numpy()))
        
        
        pers_pred = self.pred_pers_depth
        pers_pred = torch.tensor(pers_pred)[None]
        pers_pred = colormap_image(pers_pred)
        pers_pred_pil = Image.fromarray(np.uint8(255*pers_pred.detach().permute(1,2,0).cpu().numpy()))
        
        return None, pers_pred_pil


    def get_predicted_dialated_mask_viz(self):
        lookup_labels = self.mesh_generator.get_lookup_labels(torch.tensor(self.pred_topdown_mask)[:,:,0][None] == 255)
        colored_mask = self.mesh_generator.create_colors(lookup_labels, assign_colors="rainbow")
        colored_mask = colored_mask.reshape(256,256,3)
        colored_mask = Image.fromarray(np.uint8(colored_mask))
        
        return colored_mask
        
        

def colormap_image(
                image_1hw,
                mask_1hw=None, 
                invalid_color=(0.0, 0, 0.0), 
                flip=True,
                vmin=None,
                vmax=None, 
                return_vminvmax=False,
                colormap="turbo",
            ):
    """
    Colormaps a one channel tensor using a matplotlib colormap.

    Args: 
        image_1hw: the tensor to colomap.
        mask_1hw: an optional float mask where 1.0 donates valid pixels. 
        colormap: the colormap to use. Default is turbo.
        invalid_color: the color to use for invalid pixels.
        flip: should we flip the colormap? True by default.
        vmin: if provided uses this as the minimum when normalizing the tensor.
        vmax: if provided uses this as the maximum when normalizing the tensor.
            When either of vmin or vmax are None, they are computed from the 
            tensor.
        return_vminvmax: when true, returns vmin and vmax.

    Returns:
        image_cm_3hw: image of the colormapped tensor.
        vmin, vmax: returned when return_vminvmax is true.


    """
    valid_vals = image_1hw if mask_1hw is None else image_1hw[mask_1hw.bool()]
    if vmin is None:
        vmin = valid_vals.min()
    if vmax is None:
        vmax = valid_vals.max()

    cmap = torch.Tensor(
                            plt.cm.get_cmap(colormap)(
                                                torch.linspace(0, 1, 256)
                                            )[:, :3]
                        ).to(image_1hw.device)
    if flip:
        cmap = torch.flip(cmap, (0,))

    h, w = image_1hw.shape[1:]

    image_norm_1hw = (image_1hw - vmin) / (vmax - vmin)
    image_int_1hw = (torch.clamp(image_norm_1hw * 255, 0, 255)).byte().long()

    image_cm_3hw = cmap[image_int_1hw.flatten(start_dim=1)
                                        ].permute([0, 2, 1]).view([-1, h, w])

    if mask_1hw is not None:
        invalid_color = torch.Tensor(invalid_color).view(3, 1, 1).to(image_1hw.device)
        image_cm_3hw = image_cm_3hw * mask_1hw + invalid_color * (1 - mask_1hw)

    if return_vminvmax:
        return image_cm_3hw, vmin, vmax
    else:
        return image_cm_3hw