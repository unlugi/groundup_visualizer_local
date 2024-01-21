from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.jit as jit
from torch import Tensor
import open3d as o3d

class Project3D(torch.nn.Module):
    """Layer that projects 3D points into the 2D camera"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()

        self.register_buffer("eps", torch.tensor(eps).view(1, 1, 1))

    def forward(self, points_b4N: Tensor, K_b44: Tensor, cam_T_world_b44: Tensor) -> Tensor:
        """
        Projects spatial points in 3D world space to camera image space using
        the extrinsics matrix cam_T_world_b44 and intrinsics K_b44.
        """
        P_b44 = K_b44 @ cam_T_world_b44

        cam_points_b3N = P_b44[:, :3] @ points_b4N

        # from Kornia and OpenCV, https://kornia.readthedocs.io/en/latest/_modules/kornia/geometry/conversions.html#convert_points_from_homogeneous
        mask = torch.abs(cam_points_b3N[:, 2:]) > self.eps
        depth_b1N = cam_points_b3N[:, 2:] + self.eps
        scale = torch.where(mask, 1.0 / depth_b1N, torch.tensor(1.0, device=depth_b1N.device))

        pix_coords_b2N = cam_points_b3N[:, :2] * scale

        return torch.cat([pix_coords_b2N, depth_b1N], dim=1)


@torch.jit.script
def to_homogeneous(input_tensor: Tensor, dim: int = 0) -> Tensor:
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified 
    dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output_bkN = torch.cat([input_tensor, ones], dim=dim)
    return output_bkN

class BackprojectDepth(jit.ScriptModule):
    """
    Layer that projects points from 2D camera to 3D space. The 3D points are 
    represented in homogeneous coordinates.
    """

    def __init__(self, height: int, width: int):
        super().__init__()

        self.height = height
        self.width = width

        xx, yy = torch.meshgrid(
                            torch.arange(self.width), 
                            torch.arange(self.height), 
                            indexing='xy',
                        )
        pix_coords_2hw = torch.stack((xx, yy), axis=0) + 0.5

        pix_coords_13N = to_homogeneous(
                                pix_coords_2hw,
                                dim=0,
                            ).flatten(1).unsqueeze(0)

        # make these tensors into buffers so they are put on the correct GPU 
        # automatically
        self.register_buffer("pix_coords_13N", pix_coords_13N)

    @jit.script_method
    def forward(self, depth_b1hw: Tensor, invK_b44: Tensor) -> Tensor:
        """ 
        Backprojects spatial points in 2D image space to world space using 
        invK_b44 at the depths defined in depth_b1hw. 
        """
        cam_points_b3N = torch.matmul(invK_b44[:, :3, :3], self.pix_coords_13N)
        cam_points_b3N = depth_b1hw.flatten(start_dim=2) * cam_points_b3N
        cam_points_b4N = to_homogeneous(cam_points_b3N, dim=1)
        return cam_points_b4N

class SimpleVolume:
    
    """Class for housing and data handling Volumes. This class assumes
    align_corners=True for all grid_sample calls on the volume."""

    # Ensures the final voxel volume dimensions are multiples of 8
    VOX_MOD: int = 8

    def __init__(
        self,
        voxel_coords_3hwd: torch.Tensor,
        values_hwd: torch.Tensor,
        voxel_size: float,
        origin: torch.Tensor,
    ):
        """Sets internal class attributes and generates homography coordinates.
        Args:
            voxel_coords_3hwd (torch.Tensor): Tensor of shape (3, H, W, D) defining
                world coordinates of each voxel in the volume in meters.
            values_hwd (torch.Tensor): Tensor of shape (H, W, D) the volume itself.
            voxel_size (float): Size of each voxel in the volume in meters.
        """
        self.voxel_coords_3hwd = voxel_coords_3hwd
        self.values_hwd = values_hwd
        self.voxel_size = voxel_size
        self.origin = origin

        self.hom_voxel_coords_14hwd = torch.cat(
            (
                self.voxel_coords_3hwd,
                torch.ones_like(self.voxel_coords_3hwd[:1]).to(self.voxel_coords_3hwd.device),
            ),
            0,
        ).unsqueeze(0)

    @classmethod
    def from_bounds(cls, bounds: dict, voxel_size: float):
        """Creates a SimpleVolume with bounds at a specific voxel size.

        Assuming the voxel size is a multiple of VOX_MOD, the first voxel will have
        coords [xmin, ymin, zmin] and the last voxel will have coords
        [xmax, ymax, zmax] - voxel_size, so the max bounds are not included.

        X -> width
        Y -> height
        Z -> depth

        Args:
            bounds (dict): Dictionary of bounds for the volume, with keys
                xmin, xmax, ymin, ymax, zmin, zmax.
            voxel_size (float): Size of each voxel in the volume.
        Return:
            SimpleVolume: A SimpleVolume object.
        """

        expected_keys = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
        for key in expected_keys:
            if key not in bounds.keys():
                raise KeyError(
                    "Provided bounds dict need to have keys"
                    "'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'!"
                )

        # round up to the nearest multiple of VOX_MOD
        num_voxels_x = (
            int(np.ceil((bounds["xmax"] - bounds["xmin"]) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        )
        num_voxels_y = (
            int(np.ceil((bounds["ymax"] - bounds["ymin"]) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        )
        num_voxels_z = (
            int(np.ceil((bounds["zmax"] - bounds["zmin"]) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        )

        origin = torch.FloatTensor([bounds["xmin"], bounds["ymin"], bounds["zmin"]])

        voxel_coords_3hwd = cls.generate_voxel_coords_3hwd(
            origin, (num_voxels_x, num_voxels_y, num_voxels_z), voxel_size
        )

        # init to 0s
        values_hwd = torch.zeros_like(voxel_coords_3hwd[0]).float()

        return SimpleVolume(voxel_coords_3hwd, values_hwd, voxel_size, origin)

    @classmethod
    def generate_voxel_coords_3hwd(
        cls, origin: torch.Tensor, volume_dims: tuple[int, int, int], voxel_size: float
    ) -> torch.Tensor:
        """Gets world coordinates for each location in the SimpleVolume.
        Args:
            origin (torch.Tensor): Origin of the volume in world coordinates.
            volume_dims (tuple[int, int, int]): Dimensions of the volume in voxels.
            voxel_size (float): Size of each voxel in the volume.
        Returns:
            torch.Tensor: Tensor of shape (3, volume_dims[0], volume_dims[1], volume_dims[2])
                containing the world coordinates of each voxel in the volume.
        """

        grid = torch.meshgrid([torch.arange(vd) for vd in volume_dims], indexing="ij")

        voxel_coords_3hwd = origin.view(3, 1, 1, 1) + torch.stack(grid, 0) * voxel_size

        return voxel_coords_3hwd

    @classmethod
    def load(cls, filepath) -> Any:
        """Loads a SimpleVolume from disk."""
        data = np.load(filepath)

        return SimpleVolume(
            voxel_coords_3hwd=torch.tensor(data["voxel_coords_3hwd"]).float(),
            values_hwd=torch.tensor(data["values_hwd"]).float(),
            voxel_size=float(data["voxel_size"]),
            origin=torch.tensor(data["origin"]).float(),
        )

    def save(self, filepath):
        """Saves a SimpleVolume to disk."""
        np.savez_compressed(
            filepath,
            voxel_coords_3hwd=self.voxel_coords_3hwd.float().cpu().numpy(),
            values_hwd=self.values_hwd.float().cpu().numpy(),
            voxel_size=self.voxel_size,
            origin=self.origin.float().cpu().numpy(),
        )

    def cuda(self):
        """Moves SimpleVolume to gpu memory."""
        self.voxel_coords_3hwd = self.voxel_coords_3hwd.cuda()
        self.values_hwd = self.values_hwd.cuda()
        self.origin = self.origin.cuda()
        self.hom_voxel_coords_14hwd = self.hom_voxel_coords_14hwd.cuda()

    def cpu(self):
        """Moves SimpleVolume to cpu memory."""
        self.voxel_coords_3hwd = self.voxel_coords_3hwd.cpu()
        self.values_hwd = self.values_hwd.cpu()
        self.origin = self.origin.cpu()
        self.hom_voxel_coords_14hwd = self.hom_voxel_coords_14hwd.cpu()

    def to_point_cloud(self, threshold: Optional[float] = None, num_points: Optional[float] = None):
        """Converts the SimpleVolume to a point cloud with an optional threshold.

        Args:
            threshold (float): Threshold to apply to the volume before converting to a point cloud.
            num_points Optional(float): Number of points to sample from the point cloud.
        Returns:
            o3d.geometry.PointCloud: An Open3D point cloud.
        """

        if threshold is None:
            mask_hwd = torch.ones_like(self.values_hwd).bool()
        else:
            # threshold to get a mask we can use for indexing.
            mask_hwd = self.values_hwd > threshold

        # index each dim in xyz separately, then stack
        filtered_points_N3 = torch.stack(
            [voxel_dim[mask_hwd] for voxel_dim in self.voxel_coords_3hwd], 1
        )

        if num_points:
            filtered_points_N3

            perm = torch.randperm(filtered_points_N3.shape[0])
            idx = perm[:num_points]  # type: ignore

            filtered_points_N3 = filtered_points_N3[idx, :]

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(filtered_points_N3.cpu().float().numpy())

        return point_cloud

    def sample_volume(self, world_points_N3):
        """Samples the volume at world coordinates provided.
        Args:
            world_points_N3 (torch.Tensor): Tensor of shape (N, 3) containing
                world coordinates to sample the volume at.
        Returns:
            torch.Tensor: Tensor of shape (N,) containing the values of the
                volume at the provided world coordinates.
        """

        if not (world_points_N3.shape[1] == 3 and world_points_N3.ndim == 2):
            raise ValueError(
                "world_points_N3 must have shape (N, 3)! Instead got shape {}".format(
                    world_points_N3.shape
                )
            )

        world_points_N3 = world_points_N3.to(self.voxel_coords_3hwd.device)

        # convert world coordinates to voxel coordinates
        voxel_coords_N3 = world_points_N3 - self.origin.view(1, 3)
        voxel_coords_N3 = voxel_coords_N3 / self.voxel_size

        # divide by the volume_size - 1 for align corners True!
        dim_size_3 = torch.tensor(
            self.voxel_coords_3hwd.shape[1:],
            dtype=world_points_N3.dtype,
            device=world_points_N3.device,
        )
        voxel_coords_N3 = voxel_coords_N3 / (dim_size_3.view(1, 3) - 1)
        # convert from 0-1 to [-1, 1] range
        voxel_coords_N3 = voxel_coords_N3 * 2 - 1
        voxel_coords_111N3 = voxel_coords_N3[None, None, None]

        # sample the volume
        # grid_sample expects y, x, z and we have x, y, z
        # swap the axes of the coords to match the pytorch grid_sample convention
        voxel_coords_111N3 = voxel_coords_111N3[:, :, :, :, [2, 1, 0]]

        # in case we're asked to support fp16 and cpu, we need to cast to fp32 for the
        # grid_sample call
        if self.values_hwd.device == torch.device("cpu"):
            tensor_dtype = torch.float32
        else:
            tensor_dtype = self.values_hwd.dtype

        values_N = torch.nn.functional.grid_sample(
            self.values_hwd.unsqueeze(0).unsqueeze(0).type(tensor_dtype),
            voxel_coords_111N3.type(tensor_dtype),
            align_corners=True,
        ).squeeze()

        return values_N

    def project_volume_to_camera(self, cam_T_world_b44: torch.Tensor, K_b44: torch.Tensor):
        """Projects the volume to camera space."""
        batch_size = cam_T_world_b44.shape[0]

        world_points_b4N = self.hom_voxel_coords_14hwd.expand(
            batch_size, 4, *self.hom_voxel_coords_14hwd[0, 0].shape
        ).flatten(start_dim=2)

        projector = Project3D().to(world_points_b4N)
        cam_points_b3N = projector(world_points_b4N, K_b44, cam_T_world_b44)

        return cam_points_b3N


class VisibilityAggregator:
    def __init__(self, volume: SimpleVolume, additional_extent: float = 0.3):
        """
        Args:
            volume (SimpleVolume): volume to fill. Should be initialized to 0s.
            additional_extent (float): additional extent to fill beyond the
                sampled depth map. Default: 0.2 meters
        """
        self.volume = volume
        self.additional_extent = additional_extent

    def integrate_into_volume(
        self, depth_b1hw: torch.Tensor, cam_T_world_b44: torch.Tensor, K_b44: torch.Tensor
    ):
        """Fills the volume with 1s where a batch of camera frustums defined by
        cam_T_world_b44 and K_b44 are.

        Args:
            depth_b1hw (torch.tensor): depth map
            cam_T_world_b44 (torch.tensor): batch of camera poses
            K_b44 (torch.tensor): batch of **normalized** intrinsics
        """
        device = self.volume.values_hwd.device

        voxels_in_cam_b3N = self.volume.project_volume_to_camera(cam_T_world_b44, K_b44)

        vox_depth_b1N = voxels_in_cam_b3N[:, 2:3]
        # should already be 0-1 since we used normalized intrinsics
        pixel_coords_b2N = voxels_in_cam_b3N[:, :2]

        # Reshape the projected voxel coords to a 2D view of shape Hx(WxD)
        pixel_coords_bhw2 = pixel_coords_b2N.view(
            -1,
            2,
            self.volume.values_hwd.shape[0],
            self.volume.values_hwd.shape[1] * self.volume.values_hwd.shape[2],
        ).permute(0, 2, 3, 1)

        # convert to -1 to 1 for gridsample.
        pixel_coords_bhw2 = 2 * pixel_coords_bhw2 - 1

        # Sample the depth using grid sample
        sampled_depth_b1N = torch.nn.functional.grid_sample(
            input=depth_b1hw.to(device),
            grid=pixel_coords_bhw2.to(device),
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        ).flatten(2)

        valid_points_b1N = (
            (vox_depth_b1N > 1e-7)
            & (sampled_depth_b1N > 1e-7)
            & (vox_depth_b1N <= sampled_depth_b1N + self.additional_extent)
        )

        # loop through valid masks and set the volume to 1.0 for visible at
        # those locations
        for valid_points_1N in valid_points_b1N:
            # Reshape the valid mask to the volume's shape
            valid_points_hwd = valid_points_1N.view(self.volume.values_hwd.shape)
            self.volume.values_hwd[valid_points_hwd] = 1.0
            
def compute_point_cloud_metrics(
    gt_pcd: o3d.geometry.PointCloud,
    pred_pcd: o3d.geometry.PointCloud,
    max_dist: float = 1.0,
    dist_threshold: float = 0.05,
    visible_pred_indices: Optional[list[int]] = None,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute metrics for a predicted and gt point cloud.

    If the predicted point cloud is empty, all the lower-is-better metrics will be set to max_dist
    and all the higher-is-better metrics to 0.

    Args:
        gt_pcd (o3d.geometry.PointCloud): gt point cloud.
        pred_pcd (o3d.geometry.PointCloud): predicted point cloud, will be compared to gt_pcd.
        max_dist (float, optional): Maximum distance to clip distances to in meters.
            Defaults to 1.0.
        dist_threshold (float, optional): Distance threshold to use for precision
            and recall in meters. Defaults to 0.05.
        visible_pred_indices (list[int], optional): Indices of the predicted points that are
            visible in the scene. Defaults to None. When not None will be used to filter out
            predicted points when computing pred to gt.

    Returns:
        dict[str, float]: Metrics for this point cloud comparison.
    """
    metrics: Dict[str, float] = {}

    if len(pred_pcd.points) == 0:
        metrics["acc↓"] = max_dist
        metrics["compl↓"] = max_dist
        metrics["chamfer↓"] = max_dist
        metrics["precision↑"] = 0.0
        metrics["recall↑"] = 0.0
        metrics["f1_score↑"] = 0.0
        distances_pred2gt = torch.zeros([])
        distances_gt2pred = torch.zeros(len(gt_pcd.points))
        return metrics, distances_pred2gt, distances_gt2pred

    # find nearest neighbors
    distances_gt2pred = torch.tensor(gt_pcd.compute_point_cloud_distance(pred_pcd))

    # only use the visibility masks when computing pred to gt distances
    if visible_pred_indices is not None:
        pred_pcd = pred_pcd.select_by_index(visible_pred_indices)
        
    distances_pred2gt = torch.tensor(pred_pcd.compute_point_cloud_distance(gt_pcd))

    # accuracy
    metrics["acc↓"] = float(torch.mean(distances_pred2gt))

    # completion
    metrics["compl↓"] = float(torch.mean(distances_gt2pred))

    # chamfer distance
    metrics["chamfer↓"] = float(0.5 * (metrics["acc↓"] + metrics["compl↓"]))

    # precision
    metrics["precision↑"] = float((distances_pred2gt <= dist_threshold).float().mean())

    # recall
    metrics["recall↑"] = float((distances_gt2pred <= dist_threshold).float().mean())

    # F1 score
    # catch the edge case where both precision and recall are 0
    if metrics["precision↑"] + metrics["recall↑"] > 0.0:
        metrics["f1_score↑"] = (2 * metrics["precision↑"] * metrics["recall↑"]) / (
            metrics["precision↑"] + metrics["recall↑"]
        )
    else:
        metrics["f1_score↑"] = 0.0

    return metrics, distances_pred2gt, distances_gt2pred