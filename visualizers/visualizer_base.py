import os
from pathlib import Path
import pickle
import numpy as np
import torch
import PIL.Image as Image
import cv2

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class BaseVisualizer:
    def __init__(self, sample_path, dataset_root, scene_name=None):
        self.orig_image_size = (256, 256)
        self.dataset_root = dataset_root
        self.sample_idx = sample_path.split('/')[-1].split('_')[0][-4:]
        self.sample_path = sample_path
        self.scene_name = scene_name
        self.data = self.parse_path_and_read_data()
        self.cameras = self.parse_path_and_read_cameras()
        self.masks = self.parse_path_and_read_segmentation()
        
        self.pred_topdown_mask_path = "/mnt/data_s/gunlu/experiments/GroundUp/result_paper/ms_perturb_grad_normal_2/sf_test_1k_2/depth_prediction/pred_images"
        
        self.pers_pred_path = "/mnt/data_f/gunlu/Experiments/GroundUp/SimpleRecon/RESULTS/v9/epi_v9_occ_p_empty_linscale_big_ad_50_0/urbanscene3d_epipolar/sf_test_1k_2"
        self.pred_pers_depth, self.pred_pers_mask = self.load_pred_pers_pred()
        
        self.pred_topdown_mask = self.load_pred_topdown_mask()
        
        self.gt_pers_depth = self.load_gt_pers_depths()
        

    def load_pred_topdown_mask(self):
        pred_mask_path = os.path.join(self.pred_topdown_mask_path, f"mask_pred_{int(self.sample_idx)}.png")
        pred_mask = Image.open(pred_mask_path, formats=["PNG"]).convert('RGB')
        pred_mask = np.array(pred_mask)
        return pred_mask

    def parse_path_and_read_cameras(self):
        # Renderer specific
        return {'cam_perspective': None, 'cam_topdown': None,
                'K_p': None, 'K_td': None,
                'invK_p': None, 'invK_td': None}

    def parse_path_and_read_data(self):
        # Get paths for gt, proj, pred (depths) + sketch_td, sketch_p # TODO: sketch_td, sketch_p later
        path_gt = self.sample_path
        path_proj = self.sample_path.replace('gt.npy', 'proj.npy')
        path_pred = self.sample_path.replace('gt.npy', 'pred.npy')

        # Read data
        gt = np.load(path_gt)
        proj = np.load(path_proj)
        pred = np.load(path_pred)

        return {'gt': gt, 'proj': proj, 'pred': pred, "gt_perspective": self.load_gt_pers_depths()}

    def load_gt_pers_depths(self):
        perspective_depth_path = Path(self.dataset_root) / "Camera" / "depth" / "depth_{}.exr0001.exr".format(self.sample_idx)

        depth = cv2.imread(str(perspective_depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        assert depth is not None
        
        depth = depth[..., 2]
        # Get the float valid mask
        min_valid_depth=0.01
        max_valid_depth=4.0
        mask_b = ((depth >= min_valid_depth)  & (depth < max_valid_depth))
        
        # set invalids to nan or something else (sky is 6555555 in .exr files)
        depth[~mask_b] = torch.tensor(np.nan)
        
        return depth

    def load_pred_pers_pred(self):
        pickle_path = Path("/mnt/data_f/gunlu/Experiments/GroundUp/SimpleRecon/RESULTS/v9/epi_v9_occ_p_empty_linscale_big_50_0/urbanscene3d_epipolar/sf_test_1k_2/depths/")
        # pickle_path = Path("/mnt/data_f/gunlu/Experiments/GroundUp/SimpleRecon/RESULTS/v9/mono_v9_linscale_small_0/urbanscene3d_epipolar/sf_test_1k_2/depths/")
        pickle_path = pickle_path / f"{int(self.sample_idx):04d}" / f"{int(self.sample_idx):04d}.pickle"
        
        # load pickle
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['depth_pred_s0_b1hw'].squeeze(), data['foreground_pred_s0_b1hw'].squeeze()

    def parse_path_and_read_segmentation(self):
        # Get paths for cam_td, cam_p, K_td, K_p
        # path_segmap_perspective = os.path.join(self.dataset_root, 'Camera', 'segmentation','seg_test_{}.png0001.png'.format(self.sample_idx))
        path_segmap_topdown = os.path.join(self.dataset_root, 'Camera_Top_Down', 'segmentation', 'seg_test_{}.png0001.png'.format(self.sample_idx))

        # Read segmentation maps
        # segmap_perspective = Image.open(path_segmap_perspective, formats=["PNG"]).convert('RGB')
        segmap_topdown = Image.open(path_segmap_topdown, formats=["PNG"]).convert('RGB')
        segmap_topdown = np.array(segmap_topdown)

        if 'ny' in self.scene_name:  # 'train'
            ground_rgb = np.array([255, 202, 192]) / 255.
        elif 'chi' in self.scene_name:  # 'val'
            ground_rgb = np.array([254, 202, 192]) / 255.
        elif 'sf' in self.scene_name:
            ground_rgb = np.array([247, 202, 192], dtype=np.float32)  # / 255.
        else:
            ground_rgb = np.array([255, 202, 192]) / 255.

        mask_ground = (segmap_topdown[..., 0] == ground_rgb[0]) & \
                      (segmap_topdown[..., 1] == ground_rgb[1]) & \
                      (segmap_topdown[..., 2] == ground_rgb[2])

        mask_building = ~mask_ground  # buildings mask binary

        return {'mask_ground': mask_ground,
                'mask_building': mask_building,
                'segmap_topdown': segmap_topdown }
