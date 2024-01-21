import os
import numpy as np
import torch
import PIL.Image as Image

class BaseVisualizer:
    def __init__(self, sample_path, dataset_root, save_path=None, scene_name=None, sample_baseline=None, ):
        self.orig_image_size = (256, 256)
        self.dataset_root = dataset_root
        self.sample_idx = sample_path.split('/')[-1].split('_')[0][-4:]
        self.sample_path = sample_path # diffusion
        self.sample_baseline = sample_baseline # height-fields
        self.scene_name = scene_name
        self.save_path = save_path
        self.data = self.parse_path_and_read_data()
        self.cameras = self.parse_path_and_read_cameras()
        self.masks = self.parse_path_and_read_segmentation()

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
        pred_baseline = None

        if self.sample_baseline:
            path_baseline = self.sample_baseline
            pred_baseline = np.load(path_baseline)

        return {'gt': gt,
                'proj': proj,
                'pred': pred,
                'pred_baseline': pred_baseline,
                }

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
