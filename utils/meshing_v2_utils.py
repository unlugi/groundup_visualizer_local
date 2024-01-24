# TODO: this was copied directly from groundup_demo project, need to update if demo changes

import kornia
import torch
import numpy as np
import random
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import (knn_points,
                           ball_query,
                           convert_pointclouds_to_tensor)

# from app.utils.p3d_utils import (get_hom_coords_and_valid_mask_pancake_torch, get_device)
from scipy.spatial import Delaunay

import os
import time
from pathlib import Path
import numpy as np
from numba import jit
import trimesh
from skimage.morphology import binary_dilation, convex_hull_image
import torch

from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures

from utils.color_palette import bold_pastel_rainbow_palette


@jit(nopython=True)
def fast_meshing(points, H, W, hs, ws):
    faces = []
    for hh, ww in zip(hs, ws):
        # if points[1, hh, ww]:
        if points[2, hh, ww] + 1 > 0:
            idx = hh * W + ww
            if hh < (H - 1) and ww < (W - 1):
                # two triangles for each point
                if points[2, hh + 1, ww] + 1 > 0 and points[2, hh + 1, ww + 1] + 1 > 0:
                    faces.append((idx, idx + W + 1, idx + W))
                if points[2, hh, ww + 1] + 1 > 0 and points[2, hh + 1, ww + 1] + 1 > 0:
                    faces.append((idx, idx + 1, idx + 1 + W))
    return faces

class Heightfield:
    def __init__(self, origin, grid_height, grid_width, grid_resolution, cell_size=None):
        self.origin = origin  # (x, y)
        self.grid_resolution = grid_resolution
        self.heightfield = torch.ones( (grid_resolution[1], grid_resolution[0])).float()  # -torch.ones((grid_height, grid_width)).float()
        self.cell_size = cell_size
        self.grid_coords = self.get_grid_coords(grid_height, grid_width, grid_resolution)
        # labels is used to store counts, or component type (i.e. unknown, ground, object, wall)
        self.labels = torch.zeros_like(self.heightfield)

    def get_grid_coords(self, grid_height, grid_width, grid_resolution):

        self.cell_size = ( (grid_width/grid_resolution[0]), (grid_height/grid_resolution[1]) )
        x = 1 * np.arange(      0, grid_width, step=self.cell_size[0])
        y = 1 * np.arange(      0, grid_height, step=self.cell_size[0])
        # y = -1 * np.arange(-1 * grid_height, 0, step=self.cell_size[1])

        xs = self.origin[0] - x
        ys = self.origin[1] - y

        xx, yy = np.meshgrid(xs, ys)

        grid_coords = np.stack((xx, yy)) #(2xHxW)
        # grid_coords = np.vstack(np.dstack((xx, yy))) # trimesh ((H*W)x2)
        return grid_coords

    @classmethod
    def from_depth_array(cls, depth, max_scene_height, min_scene_height,
                              grid_width=4.25, grid_height=4.25, origin=None, cell_size=None):

        # Get rid of multiple channels
        if len(depth.shape) != 2:
            depth = depth[..., 2]

        # Get rid of -1 values - replace with max depth
        depth[depth == -1] = depth.max()

        # Grid resolution - square image
        grid_resolution = (depth.shape[0], depth.shape[1])

        # Grid dimensions in 3D world
        grid_width, grid_height = (grid_width, grid_height)

        # Set origin
        if origin is None:
            origin = [0.0, 0.0]
        else:
            origin = [origin[0], origin[2]]

        hh = Heightfield(origin=origin, grid_height=grid_height, grid_width=grid_width,
                         grid_resolution=grid_resolution, cell_size=cell_size)

        # Depth values will determine building and ground heights
        # hh.heightfield = 5 - depth
        hh.heightfield = depth.max() - depth

        return hh

    def to_mesh(self, rgb_values=None):

        # TODO: meshing alternative
        #  https://github.com/mikedh/trimesh/issues/1054

        # heights = self.heightfield.clone().cpu()
        heights = self.heightfield.copy()
        points = np.concatenate((self.grid_coords, heights[None,...]))

        mesh = trimesh.Trimesh()
        mesh.vertices = points[:3].reshape(3, -1).T

        faces = []
        H, W = self.grid_coords.shape[1:]
        hs, ws = np.nonzero(points[2] + 1)
        faces = fast_meshing(points, H, W, hs, ws)

        if faces:
            mesh.faces = np.vstack(faces)

        if rgb_values is not None:
            # rgb_for_mesh = np.concatenate((rgb_values, np.ones_like(rgb_values)[0:1]), 0)
            # rgb_for_mesh = (rgb_for_mesh.reshape(4, -1).T * 255).astype(np.uint8)
            rgb_for_mesh = rgb_values.astype(np.uint8)
            mesh.visual.vertex_colors = rgb_for_mesh

        # Mirror along horizontal axis and fix normals
        # mesh.vertices[:, 1] = -1 * mesh.vertices[:, 1]

        # trimesh.repair.fix_normals(mesh, multibody=False)
        # trimesh.repair.fix_winding(mesh)

        return mesh

    def to_mesh_trimesh(self, rgb_values=None):

        # Get building heights
        heights = self.heightfield.clone().cpu()

        # Flattened coordinated
        x = np.arange(0, self.grid_resolution[0]) / self.grid_resolution[0] * 4.25
        y = (self.grid_resolution[1] - np.arange(0, self.grid_resolution[1])) / self.grid_resolution[1] * 4.25
        xx, yy = np.meshgrid(x, y)

        grid_coords_flattened = np.vstack(np.dstack((-1 * yy , xx)))

        # Triangulate
        tris = Delaunay(grid_coords_flattened)

        # Mesh
        mesh = trimesh.creation.extrude_triangulation( vertices=grid_coords_flattened, faces=tris.simplices, height=-1)

        # Edit non-extruded verts with correct z values
        verts = mesh.vertices.view(np.ndarray)
        verts[verts[:, 2] == 0, 2] = 5 - heights.reshape(-1)

        return mesh

class BuildingMeshGenerator:
    def __init__(self, use_heuristic=False, cluster_heuristic='knn', apply_dilation_mask=True, building=[], use_color=False, mask_color=None):
        self.use_heuristic = use_heuristic
        self.cluster_heuristic = cluster_heuristic # ball_query or knn
        self.apply_dilation_mask = apply_dilation_mask
        self.building = building
        self.use_color = use_color
        self.mask_color = mask_color

        self.pastel_rainbow_palette = bold_pastel_rainbow_palette
        self.color_palette_idx = None

    def get_lookup_labels(self, mask_fg_tensor):

        # Apply dilation
        if self.apply_dilation_mask:
            structuring_element = torch.tensor([[1, 1],
                                                [1, 1], ], dtype=torch.float32, device=mask_fg_tensor.device)
            kernel = torch.tensor([[1, 0, 1],
                                   [0, 1, 0],
                                   [1, 0, 1]], dtype=torch.float32, device=mask_fg_tensor.device)
            mask_fg_tensor = kornia.morphology.dilation(mask_fg_tensor[None, ...], kernel=kernel)[0, ...]

        # Connected components for building mask segmentation
        mask_fg_labels = kornia.contrib.connected_components(mask_fg_tensor[None, ...].float(), num_iterations=400)

        lookup_labels = mask_fg_labels.detach().cpu().numpy()[0, 0]

        return lookup_labels

    def create_colors(self, lookup_labels, assign_colors='random'):
        label_colors = lookup_labels / lookup_labels.max()
        idx_building = label_colors > -1  # label_colors != 0.0
        label_c = label_colors[idx_building]

        # Gradient Effect
        if assign_colors == 'gradient':
            label_c_rgb = np.hstack((label_c[:,None],
                                    0.2*np.ones((label_c.shape[0], 1)),
                                    0.4*np.ones((label_c.shape[0], 1))))

        # Grayscale Gradient Effect
        elif assign_colors == 'grayscale':
            label_c_rgb = np.tile(label_c[..., None], (1, 3))

        # Random Color Effect
        elif assign_colors == 'random':
            unique_labels, reverse_indices = np.unique(label_c, return_inverse=True)
            label_rgb_unique = np.random.randint(low=180,
                                                 high=255,
                                                 size=[len(unique_labels), 3],
                                                 dtype=np.uint8)  # low=0 for bold colors
            label_c_rgb = np.ones((label_c.shape[0], 3))
            for idx in range(len(unique_labels)):
                if idx == 0:  # ground is always green # rgb(211, 211, 211) grey
                    # label_c_rgb[reverse_indices == idx] = np.array([98, 227, 132], dtype=np.uint8)
                    label_c_rgb[reverse_indices == idx] = np.array([250,250,250], dtype=np.uint8)
                else:
                    label_c_rgb[reverse_indices == idx] = label_rgb_unique[idx]
        elif assign_colors == 'gt':
            label_c_rgb = self.mask_color['segmap_topdown'][0].view(-1, 3)
            label_c_rgb = label_c_rgb.cpu().numpy()

        elif assign_colors == 'rainbow':

            # Get unique labels for each detected building
            unique_labels, reverse_indices = np.unique(label_c, return_inverse=True)

            # Generate a random color index for each building from the color palette
            if self.color_palette_idx is None:
                idx_palette = np.random.randint(low=0, high=len(self.pastel_rainbow_palette), size=len(unique_labels))
                # idx_palette = random.sample(range(0, len(self.pastel_rainbow_palette)), len(unique_labels))
                self.color_palette_idx = idx_palette
            else:
                idx_palette = self.color_palette_idx

            label_c_rgb = np.ones((label_c.shape[0], 3))
            for idx in range(len(unique_labels)):
                if idx == 0:  # ground is always green # rgb(211, 211, 211) grey
                    # label_c_rgb[reverse_indices == idx] = np.array([98, 227, 132], dtype=np.uint8)
                    label_c_rgb[reverse_indices == idx] = np.array([250,250,250], dtype=np.uint8)
                else:
                    label_c_rgb[reverse_indices == idx] = self.pastel_rainbow_palette[idx_palette[idx]]


        else:
            print('unrecognized color assignment!')

        label_colors = label_c_rgb
        return label_colors

    def generate_mesh(self, depths, origin=None, grid_size=None, ):

        color_per_building_n3 = None

        if self.use_color and self.mask_color is not None:
            # Create per-building labels
            building_lookup_labels = self.get_lookup_labels(self.mask_color['mask_building'])

            # Create per-building colors
            # color_per_building_n3 = self.create_colors(building_lookup_labels, assign_colors='gt').astype(np.uint8)
            # color_per_building_n3 = self.create_colors(building_lookup_labels, assign_colors='random').astype(np.uint8)
            color_per_building_n3 = self.create_colors(building_lookup_labels, assign_colors='rainbow').astype(np.uint8)

        if origin is None and grid_size is None:
            heightfield_buildings = Heightfield.from_depth_array(depths, min_scene_height=0.01, max_scene_height=5.0)
        else:
            heightfield_buildings = Heightfield.from_depth_array(depths,
                                                                 origin=origin,
                                                                 grid_height=grid_size[0], grid_width=grid_size[1],
                                                                 min_scene_height=0.01, max_scene_height=5.0)
        meshed_buildings = heightfield_buildings.to_mesh(rgb_values=color_per_building_n3)

        return meshed_buildings

    def get_pointcloud(self, coords_hom_uv1, sample_data, camera_pose, foreground_mask):
        # Get device
        device = get_device()

        # Get camera intrinsics and extrinsics
        Kinv_td = camera_pose['Kinv_td']
        extrinsics_RT_td = camera_pose['camera_RT_pc_td']

        # Backproject to 3d
        coords_hom_uv1 = torch.from_numpy(coords_hom_uv1).type(torch.float32).to(device)
        coords_cam = Kinv_td @ coords_hom_uv1.T
        pts3D_topdown = extrinsics_RT_td.inverse() @ coords_cam

        pts3D_topdown = pts3D_topdown.type(torch.float32)

        pts3D_topdown_ = pts3D_topdown[:3, :].view(3, 256, 256)
        pts3D_topdown_ = pts3D_topdown_[:, foreground_mask]

        point_cloud_td = Pointclouds(points=pts3D_topdown_[:3, :].T[None, ...])
        return point_cloud_td

    def run(self, sample_topdown, return_point_cloud=False, sample_data=None, output=None, camera_pose=None):
        if self.use_heuristic:
            return self.run_meshing_heuristic(sample_data, output, camera_pose)
        else:
            return self.run_meshing(sample_topdown, return_point_cloud=return_point_cloud)

    def run_meshing(self, sample_topdown, return_point_cloud):
        # Run meshing algorithm - elevation
        mesh_sample_topdown = self.generate_mesh(sample_topdown)
        if return_point_cloud:
            points = mesh_sample_topdown.vertices
            pc_sample_topdown = trimesh.PointCloud(points)
            return pc_sample_topdown
        else:
            return mesh_sample_topdown

    def run_meshing_heuristic(self, sample_data, output, camera_pose):
        ...
        # Get building label masks via connected components labelling from topdown masks
        mask_fg_labels = kornia.contrib.connected_components(output['predicted_mask_topdown'].unsqueeze(0), num_iterations=200)
        lookup_labels = mask_fg_labels.detach().cpu().numpy()[0, 0]

        # Get pancake point-clouds from topdown predictions
        depth_topdown = output['raw_topdown_depthmap'].squeeze(0).squeeze(0).detach().cpu().numpy()

        # Clean this part up after torch implementation for pancake is ready
        # Remove negative depths
        neg = depth_topdown < 0
        depth_topdown_2 = depth_topdown.copy()
        depth_topdown_2[neg] = 5.0

        depth_td_2 = depth_topdown_2 - 5 + 100

        # Get homogeneous image coordinates
        coords_hom_uv1, mask = get_hom_coords_and_valid_mask_pancake_torch(depth_td_2, threshold=150)

        # Label point cloud
        # Get building pixel mask
        label_colors = lookup_labels / lookup_labels.max()
        idx_building = label_colors != 0.0

        # Get pointclouds
        point_cloud_label = self.get_pointcloud(coords_hom_uv1, sample_data, camera_pose, idx_building)

        # Depth prediction pointcloud
        mask_buil = depth_topdown_2 < 4.80
        point_cloud_td_depth = self.get_pointcloud(coords_hom_uv1, sample_data, camera_pose, mask_buil)

        # Assign heights to each building region via knn from topdown depth predictions
        if self.cluster_heuristic == 'knn':
            nn_dists, nn_idx, nn = knn_points(convert_pointclouds_to_tensor(point_cloud_label)[0],
                                              convert_pointclouds_to_tensor(point_cloud_td_depth)[0],
                                              K=20)
        elif self.cluster_heuristic == 'ball_query':
            nn_dists, nn_idx, nn = ball_query(convert_pointclouds_to_tensor(point_cloud_label)[0],
                                              convert_pointclouds_to_tensor(point_cloud_td_depth)[0],
                                              K=20,
                                              radius=0.15)
        else:
            print('unrecognized cluster heuristic!')

        ####
        # for each point in label mask image ~8K
        # get kkn points from depth
        # calculate mean depth value from the neighbors
        # assign that mean depth to depths_init_knn
        # decide what you want to do with building mask regions - heuristic pick tallest observation assign to all
        val_depth = depth_topdown_2[mask_buil]

        idx_pc = 0
        knn_idx = nn_idx[idx_pc]

        depths_init_knn = np.full_like(mask_buil, fill_value=5.0, dtype=float)
        depth_init_vec = np.zeros(shape=knn_idx.shape[0], dtype=float)

        for point_idx in range(knn_idx.shape[0]):
            current_idx = knn_idx[point_idx]

            # cast current_idx to numpy
            current_idx_np = current_idx.detach().cpu().numpy()

            curr_depths = val_depth[current_idx_np]
            depth_init_vec[point_idx] = np.median(curr_depths)  # curr_depths.mean() # np.median(curr_depths)

        depths_init_knn[idx_building] = depth_init_vec

        # (Optional) Select mean-median-min depth for each building region
        # Create initial_depths
        depths_floodfilled = np.zeros_like(lookup_labels) + 5.0
        for building_id in np.unique(lookup_labels):
            # print(building_id)
            building_mask = lookup_labels == building_id
            building_depths = depths_init_knn[building_mask]

            building_depths_valid = building_depths[building_depths >= 0]
            # print(building_mask.shape)
            # print(building_depths_valid.shape)

            if len(building_depths_valid) == 0:
                continue

            if building_id == 0:
                depths_floodfilled[building_mask] = 5.0
                continue

            # depths_floodfilled[building_mask] = building_depths_valid.mean()
            depths_floodfilled[building_mask] = building_depths_valid.min()

        # Run meshing algorithm
        mesh_buildings = self.generate_mesh(depths_floodfilled)
        mesh_buildings_knn = self.generate_mesh(depths_init_knn)

        # Return mesh
        return mesh_buildings, mesh_buildings_knn



def update_vertex_colors(mesh, target_color):
    """
    Update vertex colors of the mesh based on a specific target color.

    Parameters:
    - mesh (Meshes): PyTorch3D Meshes object.
    - target_color (np.ndarray): Target color array of shape (3,).

    Returns:
    - Meshes: Updated Meshes object.
    """
    # Get faces and vertex indices
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    # normals = mesh.faces_normals_packed()
    vertex_colors = mesh.textures._verts_features_padded.squeeze(0)
    vertex_colors_new = vertex_colors.clone()

    # convert target color to tensor
    target_color = torch.tensor(target_color, device=vertex_colors.device, dtype=torch.float32)

    # Calculate the tolerance based on the specified decimal points
    tolerance = 10 ** (-4)

    # TODO: to speed-up, get the indices of vertices with the target color - then only loop through those faces
    # Identify faces with the target color
    for face in faces:
        current_vertex_colors = vertex_colors[face]

        comparison_result = torch.isclose(current_vertex_colors, target_color, rtol=tolerance, atol=tolerance).all(dim=1)

        # indices of different colors
        idx_to_get_color = torch.nonzero(~comparison_result)
        idx_to_change_color = torch.nonzero(comparison_result)

        if len(idx_to_get_color) != 0:
            # update vertex colors
            for idx in idx_to_change_color:
                vertex_colors_new[face[idx]] = vertex_colors[face[idx_to_get_color[0]]]

    # Create a new Meshes object with updated vertex colors
    textures = Textures(verts_rgb=vertex_colors_new[None, ...])
    # updated_mesh = Meshes(verts=verts, faces=faces, textures=Meshes.textures(verts_rgb=vertex_colors_new))
    updated_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    return updated_mesh


def update_vertex_colors_fast(mesh, target_color):
    """
    Update vertex colors of the mesh based on a specific target color.

    Parameters:
    - mesh (Meshes): PyTorch3D Meshes object.
    - target_color (np.ndarray): Target color array of shape (3,).

    Returns:
    - Meshes: Updated Meshes object.
    """
    # Get faces and vertex indices
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    normals = mesh.faces_normals_padded()
    vertex_colors = mesh.textures._verts_features_padded.squeeze(0)
    vertex_colors_new = vertex_colors.clone()

    # Convert target color to tensor
    target_color = torch.tensor(target_color, device=vertex_colors.device, dtype=torch.float32)

    # Calculate the tolerance based on the specified decimal points
    tolerance = 10 ** (-4)

    # Identify vertices with the target color
    target_vertex_mask = torch.isclose(vertex_colors, target_color, rtol=tolerance, atol=tolerance).all(dim=1)

    if target_vertex_mask.any():
        # Get the indices of vertices with the target color
        target_vertex_indices = torch.nonzero(target_vertex_mask).squeeze()

        faces_target = []
        for face in faces:
            comparison = (face[:,None] == target_vertex_indices).any(dim=1)
            if comparison.all():
                continue
            elif (~comparison).all():
                continue
            else:
                faces_target.append(face)

        # Loop over the subset of faces with target vertices on GPU # TODO: how to get rid of this loop?
        for face in faces_target:
            current_vertex_colors = vertex_colors[face]

            comparison_result = torch.isclose(current_vertex_colors, target_color, rtol=tolerance, atol=tolerance).all(dim=1)

            # Indices of different colors
            idx_to_get_color = torch.nonzero(~comparison_result)
            idx_to_change_color = torch.nonzero(comparison_result)

            if len(idx_to_get_color) != 0:
                # Update vertex colors on GPU
                for idx in idx_to_change_color:
                    vertex_colors_new[face[idx]] = vertex_colors[face[idx_to_get_color[0]]]

    # Create a new Meshes object with updated vertex colors
    textures = Textures(verts_rgb=vertex_colors_new[None, ...])
    # updated_mesh = Meshes(verts=verts, faces=faces, textures=Meshes.textures(verts_rgb=vertex_colors_new))
    updated_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    return updated_mesh



def update_vertex_colors_fast_padding(mesh, target_color):
    """
    Update vertex colors of the mesh based on a specific target color.

    Parameters:
    - mesh (Meshes): PyTorch3D Meshes object.
    - target_color (np.ndarray): Target color array of shape (3,).

    Returns:
    - Meshes: Updated Meshes object.
    """
    # Get faces and vertex indices
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    # normals = mesh.faces_normals_packed()
    vertex_colors = mesh.textures._verts_features_padded.squeeze(0)
    vertex_colors_new = vertex_colors.clone()

    # Convert target color to tensor
    target_color = torch.tensor(target_color, device=vertex_colors.device, dtype=torch.float32)

    # Calculate the tolerance based on the specified decimal points
    tolerance = 10 ** (-4)

    # Identify vertices with the target color
    target_vertex_mask = torch.isclose(vertex_colors, target_color, rtol=tolerance, atol=tolerance).all(dim=1)

    if target_vertex_mask.any():
        # Get the indices of vertices with the target color
        target_vertex_indices = torch.nonzero(target_vertex_mask).squeeze()

        # Get length of arrays
        len_faces = faces.shape[0]
        len_target_vertex_indices = target_vertex_indices.shape[0]

        faces_appended = faces.clone()
        if len_faces > len_target_vertex_indices:
            # Pad target_vertex_indices with -1
            to_pad = -1 * torch.ones(np.abs(len_faces - len_target_vertex_indices),
                                     device=target_vertex_indices.device,
                                     dtype=torch.int64)
            target_vertex_indices = torch.cat((target_vertex_indices, to_pad), dim=0)

        elif len_faces < len_target_vertex_indices:
            # Pad faces with -1
            to_pad = -1 * torch.ones(np.abs(len_target_vertex_indices - len_faces),
                                     faces_appended.shape[1],
                                     device=faces.device,
                                     dtype=torch.int64)
            faces_appended = torch.cat((faces_appended, to_pad), dim=0)
        else:
            print('No padding needed!')

        faces_mask = (faces_appended.unsqueeze(2) == target_vertex_indices).any(dim=1)

        # Loop over the subset of faces with target vertices on GPU
        for face in faces_appended[faces_mask]:
            current_vertex_colors = vertex_colors[face]

            comparison_result = torch.isclose(current_vertex_colors, target_color, rtol=tolerance, atol=tolerance).all(dim=1)

            # Indices of different colors
            idx_to_get_color = torch.nonzero(~comparison_result)
            idx_to_change_color = torch.nonzero(comparison_result)

            if len(idx_to_get_color) != 0:
                # Update vertex colors on GPU
                for idx in idx_to_change_color:
                    vertex_colors_new[face[idx]] = vertex_colors[face[idx_to_get_color[0]]]

    # Create a new Meshes object with updated vertex colors
    textures = Textures(verts_rgb=vertex_colors_new[None, ...])
    # updated_mesh = Meshes(verts=verts, faces=faces, textures=Meshes.textures(verts_rgb=vertex_colors_new))
    updated_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    return updated_mesh



def get_xy_depth_homogeneous_coordinates_bs1_vis(depth_map, foreground_mask):
    # Get foreground pixels xy and values depth_pixels_2d_homogeneous

    depth_pixels_values = depth_map[foreground_mask][:, None]
    depth_pixels_xy = torch.nonzero(~torch.isnan(depth_map), as_tuple=True)[1:]
    depth_pixels_xy = torch.stack(list(depth_pixels_xy), dim=1)

    # depth_pixels_2d_homogeneous = [z*x z*y z*1 1]
    depth_pixels_2d_homogeneous = torch.concat( (depth_pixels_values * depth_pixels_xy,
                                                 depth_pixels_values,
                                                 torch.ones_like(depth_pixels_values)), dim=1)

    return depth_pixels_2d_homogeneous, foreground_mask



