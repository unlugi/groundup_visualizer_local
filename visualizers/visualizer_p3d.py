import os
import time
import numpy as np
import torch
import PIL.Image as Image
import trimesh
from tqdm import tqdm
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO

from trimesh.exchange.obj import export_obj

from .visualizer_base import BaseVisualizer
from utils.camera_utils import load_camera
from utils.meshing_v2_utils import BuildingMeshGenerator, \
                                   update_vertex_colors, update_vertex_colors_fast, update_vertex_colors_fast_padding, \
                                   get_xy_depth_homogeneous_coordinates_bs1_vis
from utils.p3d_utils import define_camera, mesh_renderer, point_cloud_renderer

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

class GroundUpVisualizerP3D(BaseVisualizer):
    def __init__(self, sample_path, dataset_root, save_path, scene_name, samples_baseline, add_color_to_mesh=None, device='cpu'):
        super().__init__(sample_path, dataset_root, save_path, scene_name,samples_baseline)
        self.device = self.get_device(device)
        self.masks = self.move_to_device(self.masks)
        self.cameras = self.parse_path_and_read_cameras()
        self.add_color_to_mesh = add_color_to_mesh
        self.mesh_generator = BuildingMeshGenerator(use_color=add_color_to_mesh, mask_color=self.masks, apply_dilation_mask=False)
        self.mesh_dict = {
            # 'gt': None,
                          'pred': None,
                          'pred_baseline': None,
                          # 'pc_gt': None,
                          # 'pc_pred_proj': None,
                          }

    def get_device(self, device_name):
        # Set device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = '0' if device_name=='cuda' else '-1'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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



    def parse_path_and_read_data(self):
        # Get paths for gt, proj, pred (depths) + sketch_td, sketch_p # TODO: sketch_td, sketch_p later
        path_gt = self.sample_path
        path_proj = self.sample_path.replace('gt.npy', 'proj.npy')
        path_pred = self.sample_path.replace('gt.npy', 'pred.npy')

        # Read data
        gt = np.load(path_gt)
        proj = np.load(path_proj)
        pred = np.load(path_pred)
        pred_baseline = None

        if self.sample_baseline:
            path_baseline = self.sample_baseline
            pred_baseline = np.load(path_baseline)


        # Hacking perspective depths
        path_gt_perspective = os.path.join(self.dataset_root, 'Camera', 'depth',
                                        "depth_{}.exr0001.exr".format(self.sample_idx))

        gt_perspective = cv2.imread(path_gt_perspective, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 2]


        return {'gt': gt,
                'proj': proj,
                'pred': pred,
                'pred_baseline': pred_baseline,
                'gt_p': gt_perspective,
                }


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

        print('Generating mesh...')
        
        threshold = 10000
        depth_pixels_values = self.data[mode][self.data[mode] < threshold][:,None]

        # offset if ground too low
        max_depth = self.data[mode].max()
        offset_ground = self.data['gt'].max() - self.data[mode].max()

        depth_pixels_values = depth_pixels_values + offset_ground + (100.0 - 5.0) # TODO: hard-coded # TODO: BUG

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
        elif model_name == 'pred_baseline':
            mesh = self.mesh_generator.generate_mesh(depths=self.data['pred_baseline'],
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
        # self.mesh_dict[model_name] = Meshes(verts=[verts_transformed], faces=[faces], verts_normals=[normals], textures=textures)



        if update_face_colors:
            print("Fixing mesh colors...")
            # target_color_value = np.array([98, 227, 132]) / 255.0 #
            target_color_value = np.array([250,250,250]) / 255.0

            start = time.time()
            # Update vertex colors based on the target color
            self.mesh = update_vertex_colors_fast(self.mesh, target_color_value)

            end = time.time()
            print('time it took:{} seconds '.format(end-start))

        # Update mesh in mesh_dict
        self.mesh_dict[model_name] = self.mesh # TODO: check this out


    def render_scene(self, image_size=(256, 256), offset=(0, 0, 0)):

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

        # TODO: fix this
        # self.cameras_perspective_p3d = cameras_perspective.clone()

        # Render the scene
        renderer_ = mesh_renderer( cameras=cameras_perspective, imsize=image_size, device=self.device, offset=offset)
        perspective_render = renderer_(self.mesh)
        # perspective_render = renderer_(self.mesh_dict[mode])


        perspective_color = perspective_render[0, :].detach().cpu().numpy()
        perspective_color = Image.fromarray((perspective_color * 255.0).astype(np.uint8))
        perspective_color = perspective_color.convert('RGBA')

        background_color = (255, 255, 255, 255)  # RGBA value for white
        image_w_bg = Image.new("RGBA", perspective_color.size, background_color)

        # Copy RGB values from the original image and set alpha to 255 (fully opaque)
        image_w_bg.paste(perspective_color, (0, 0), mask=perspective_color)

        return image_w_bg


    def export_mesh_p3d(self, mesh_name, save_path, update_face_colors=True, mode='gt', export_p3d=False):

        mesh_pytorch3d = self.mesh
        # mesh_pytorch3d = self.mesh_dict[mode] # TODO: check this out

        if update_face_colors:
            print("Fixing mesh colors...")

            # target_color_value = np.array([98, 227, 132])/ 255.0 # 211, 211, 211
            target_color_value = np.array([250,250,250]) / 255.0

            # Update vertex colors based on the target color
            mesh_pytorch3d = update_vertex_colors_fast(mesh_pytorch3d, target_color_value)

        # file save path
        filename = os.path.join(save_path, mesh_name)

        if export_p3d:
            IO().save_mesh(mesh_pytorch3d, filename+".obj", include_textures=True)

        else: # trimesh
            # Convert PyTorch3D mesh to trimesh
            verts_np = mesh_pytorch3d.verts_packed().detach().cpu().numpy()
            faces_np = mesh_pytorch3d.faces_packed().detach().cpu().numpy()
            vertex_colors_np = mesh_pytorch3d.textures._verts_features_padded.detach().cpu().numpy()
            vertex_colors_np =(vertex_colors_np * 255.0).astype(np.uint8)

            mesh_trimesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, vertex_colors=vertex_colors_np)
            mesh_trimesh.visual.vertex_colors = vertex_colors_np
            trimesh.repair.fix_winding(mesh_trimesh)
            trimesh.repair.fix_inversion(mesh_trimesh, multibody=False)
            trimesh.repair.fix_normals(mesh_trimesh, multibody=False)

            mesh_trimesh.export(filename+".obj", file_type='obj', include_color=True)
            # mesh_with_color = export_obj(mesh_trimesh, include_color=True, include_normals=True)
            # mesh_with_color.export(filename, file_type='obj', include_color=True)


    def get_pointcloud_depth_perspective_in_world_coordinates(self, mode='gt', is_save=False,
                                                              path_to_save=None, minmax_valid_depth=(0.01, 4.0)):

        # Get correct depth map
        if 'proj' in mode:
            depth_map = self.data['proj'].copy() # depth offset
            # Get the float valid mask
            minmax_valid_depth = (0.01, 6)
            mask_for_valid_depth = ((depth_map >= minmax_valid_depth[0])  # change valid_depth
                                    & (depth_map < minmax_valid_depth[1]))
            # offset ortho2perspective # offset if ground too low
            max_depth = self.data['proj'].max()
            offset_ground = self.data['gt'].max() - self.data['proj'].max()

            depth_map = depth_map + offset_ground + (100.0 - 5.0)  # TODO: BUG
        else:
            depth_map = self.data['gt_p'].copy()
            # depth_map = self.data['gt_perspective'].copy()
            # Get the float valid mask
            minmax_valid_depth = (0.01, 4.0)
            mask_for_valid_depth = ((depth_map >= minmax_valid_depth[0])  # change valid_depth
                                    & (depth_map < minmax_valid_depth[1]))

        # Get correct camera intrinsics and extrinsics
        if 'proj' in mode:
            Kinv = self.cameras['invK_td'].copy()
            Kinv = torch.from_numpy(Kinv).type(torch.float32).to(self.device)
            extrinsics_RT = self.cameras['cam_topdown']['camera_pc'].copy()  # world to camera
            extrinsics_RT = torch.from_numpy(extrinsics_RT).type(torch.float32).to(self.device)
        else:
            Kinv = self.cameras['invK_p'].copy()
            Kinv = torch.from_numpy(Kinv).type(torch.float32).to(self.device)
            extrinsics_RT = self.cameras['cam_perspective']['camera_pc'].copy()  # world to camera
            extrinsics_RT = torch.from_numpy(extrinsics_RT).type(torch.float32).to(self.device)

        # Convert to torch tensor
        depth_map = torch.from_numpy(depth_map).to(self.device)
        mask_for_valid_depth = torch.from_numpy(mask_for_valid_depth).to(self.device)

        # Mask out invalid depth values
        depth_map[~mask_for_valid_depth] = torch.tensor(np.nan)

        # UV coordinates of the depth map - [u*1, v*d, d, 1] - 2D homogeneous coordinates from depth pixels
        coords_homogeneous_uv1, foreground_mask = get_xy_depth_homogeneous_coordinates_bs1_vis(depth_map[None, ...],
                                                                                      mask_for_valid_depth[None, ...],
                                                                                      )

        # # Get camera intrinsics and extrinsics
        # Kinv = self.cameras['invK_p'].copy()
        # Kinv = torch.from_numpy(Kinv).type(torch.float32).to(self.device)
        # extrinsics_RT = self.cameras['cam_perspective']['camera_pc'].copy() # world to camera
        # extrinsics_RT = torch.from_numpy(extrinsics_RT).type(torch.float32).to(self.device)

        # Backproject to 3d
        coords_cam = Kinv @ coords_homogeneous_uv1.T
        pts3D = extrinsics_RT.inverse() @ coords_cam

        # Colors
        # colors = 0.3 * torch.ones_like((coords_homogeneous_uv1))[..., :3]
        # colors = torch.ones_like(coords_homogeneous_uv1[..., :3]) * torch.tensor([1.0, 0.0, 0.0]).to(self.device)
        colors = torch.ones_like(coords_homogeneous_uv1[..., :4]) * torch.tensor([1.0, 0.0, 0.0, 1.0]).to(self.device)
        point_cloud = Pointclouds(points=pts3D[:3, :].T[None, ...], features=colors[None, ...])
        # point_cloud = Pointclouds(points=pts3D[:3, :].T[None, ...])


        if is_save:
            filename_pc = os.path.join(path_to_save, '{0}_{1}.ply'.format(mode, self.sample_idx))
            IO().save_pointcloud(point_cloud, filename_pc, colors_as_uint8=False)

        self.mesh_dict[mode] = point_cloud

    def render_pointcloud(self, image_size=(256, 256), offset=(0, 0, 0), mode='gt'):
        renderer_pointcloud = point_cloud_renderer(self.cameras_perspective_p3d, image_size, is_depth=False).to(self.device)

        # image_rendered = renderer_pointcloud(self.mesh_dict[mode])
        image_rendered = renderer_pointcloud(self.mesh_dict[mode], gamma=(1e-4,),
                          bg_col=torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device=self.device),
                                             znear=[0.01,],
                                                zfar=[4.0,],)

                                             # znear=torch.tensor([0.01]).to(self.device),
                                             # zfar=torch.tensor([4.0]).to(self.device))

        image_pc = image_rendered[0, :].detach().cpu().numpy()
        image_pc = Image.fromarray((image_pc * 255).astype(np.uint8))
        image_pc= image_pc.convert('RGBA')

        return image_pc



    def mesh_and_render_all_modes(self, image_size=(256, 256),
                                        light_offset=(-3.0, 0.0, 4.0),
                                        render_scene=False,
                                        export_mesh=False,
                                        fix_colors=False):

        # Create save path for current scene
        save_path = os.path.join(self.save_path, self.sample_idx)
        print(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for mode in tqdm(self.mesh_dict.keys(), desc='Meshing Results', unit='mode'):

            time.sleep(1)  # Simulating some processing time
            print('\nCurrent mode: ', mode)

            if mode == 'pc':
                print('Skipping pointcloud mode')
                time.sleep(1)  # Simulating some processing time
                print('-' * 100)
                time.sleep(1)  # Simulating some processing time
                continue

            # Get the mesh
            print('Obtaining 3D mesh...')
            self.get_mesh_in_world_coordinates(mode, update_face_colors=fix_colors)

            if render_scene:
                print('Rendering scene...')
                # Render the scene with the current mesh mode
                rendered_image = self.render_scene(image_size=image_size, offset=light_offset)
                # Save the rendered image
                path_render = os.path.join(save_path, "render_{}_{}.png".format(mode, self.sample_idx))
                rendered_image.save(path_render, 'PNG')

            if export_mesh:
                print('Exporting mesh...')
                self.export_mesh_p3d(mesh_name='mesh_{}'.format(mode), save_path=save_path, update_face_colors=False)

            time.sleep(0.1)  # Simulating some processing time
            print('-' * 100)
            time.sleep(0.1)  # Simulating some processing time



    # def mesh_and_render_all_modes(self, image_size=(256, 256),
    #                                     light_offset=(-3.0, 0.0, 4.0),
    #                                     render_scene=False,
    #                                     export_mesh=False,
    #                                     fix_colors=False):
    #
    #     # Create save path for current scene
    #     save_path = os.path.join(self.save_path, self.sample_idx)
    #     print(save_path)
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #     for mode in tqdm(self.mesh_dict.keys(), desc='Meshing Results', unit='mode'):
    #
    #         time.sleep(1)  # Simulating some processing time
    #         print('\nCurrent mode: ', mode)
    #
    #         if 'pc' in mode:
    #             print('Generating 3D pointcloud...')
    #             self.get_pointcloud_depth_perspective_in_world_coordinates(mode=mode,
    #                                                                        is_save=False,
    #                                                                        path_to_save=save_path)
    #             print('Rendering 3D pointcloud...')
    #             rendered_image_pointcloud = self.render_pointcloud(image_size=image_size, offset=light_offset, mode=mode)
    #             # Save the rendered image
    #             path_render_pc = os.path.join(save_path, "render_{}_{}.png".format(mode, self.sample_idx))
    #             rendered_image_pointcloud.save(path_render_pc, 'PNG')
    #             continue
    #
    #         # Get the mesh
    #         print('Obtaining 3D mesh...')
    #         self.get_mesh_in_world_coordinates(mode, update_face_colors=fix_colors)
    #
    #         if render_scene:
    #             print('Rendering scene...')
    #             # Render the scene with the current mesh mode
    #             rendered_image = self.render_scene(image_size=image_size, offset=light_offset)
    #             # Save the rendered image
    #             path_render = os.path.join(save_path, "render_{}_{}.png".format(mode, self.sample_idx))
    #             rendered_image.save(path_render, 'PNG')
    #
    #         if export_mesh:
    #             print('Exporting mesh...')
    #             self.export_mesh_p3d(mesh_name='mesh_{}'.format(mode), save_path=save_path, update_face_colors=False, export_p3d=True)
    #
    #         time.sleep(0.1)  # Simulating some processing time
    #         print('-' * 100)
    #         time.sleep(0.1)  # Simulating some processing time

