import glob
import json
import os
import time
from pathlib import Path

import click
import numpy as np
import open3d as o3d
import torch
from pytorch3d.io import IO
from tqdm import tqdm

from utils.mesh_metrics_utils import (SimpleVolume, VisibilityAggregator,
                                      compute_point_cloud_metrics, BackprojectDepth)
from visualizers.visualizer_p3d import \
    GroundUpVisualizerP3D as GroundUpVisualizer

"""
Run with something like:
CUDA_VISIBLE_DEVICES=3 python compute_mesh_metrics.py \
--save_path debug_dump/ \
--predictions_path /mnt/data_f/gunlu/Experiments/GroundUp/Diffusion/diff_results_v14/test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3/test/epoch0035/ \
--save_path debug_dump/test231202_235909_v14_guide_gt_pt_rn_pt_vae_l1l2_max5.2_lr3

If you need to output debug renders and meshes, use the --debug_dump flag. Use --verbose to print scenes metrics and logs to console.
"""

def get_visualizer(sample_path, dataset_root, mode="gt", scene_name=None, verbose=False):
    add_mesh_color = True

    gup_visualizer = GroundUpVisualizer(
        sample_path=sample_path,
        dataset_root=dataset_root,
        scene_name=scene_name,
        add_color_to_mesh=add_mesh_color,
        device='cuda',
        verbose=verbose,
    )
    # Run meshing
    # gup_visualizer.get_mesh_in_world_coordinates(model_name=mode)
    
    return gup_visualizer


def dump_debug_viz(gup_visualizer, save_path, mode="gt", offset=(-3.0, 0.0, 4.0)):
    # Save mesh
    filename = os.path.join(save_path, "mesh_{}_elevation_{}.ply".format(mode, gup_visualizer.sample_idx))
    IO().save_mesh(gup_visualizer.mesh, filename)
    
    # Render image
    cam_p_p3d, rendered_image = gup_visualizer.render_scene(image_size=(1024, 1024), offset=offset)
    filename = os.path.join(save_path, "image_{}_{}.png".format(mode, gup_visualizer.sample_idx))
    rendered_image.save(filename, 'PNG')
    return cam_p_p3d, rendered_image


def compute_depth_metrics_batched(gt_bN, pred_bN, valid_masks_bN, mult_a=False):
    """
    Computes error metrics between predicted and ground truth depths, 
    batched. Abuses nan behavior in torch.
    """

    gt_bN = gt_bN.clone()
    pred_bN = pred_bN.clone()

    gt_bN[~valid_masks_bN] = torch.nan
    pred_bN[~valid_masks_bN] = torch.nan

    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a_dict = {}
    
    a_val = (thresh_bN < (1.0+0.05)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a5"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a10"] = torch.nanmean(a_val, dim=1) 

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a25"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.10)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a0"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25)     ).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a1"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 2).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a2"] = torch.nanmean(a_val, dim=1)

    a_val = (thresh_bN < (1.0+0.25) ** 3).float()
    a_val[~valid_masks_bN] = torch.nan
    a_dict[f"a3"] = torch.nanmean(a_val, dim=1)

    if mult_a:
        for key in a_dict:
            a_dict[key] = a_dict[key]*100

    rmse_bN = (gt_bN - pred_bN) ** 2
    rmse_b = torch.sqrt(torch.nanmean(rmse_bN, dim=1))

    rmse_log_bN = (torch.log(gt_bN) - torch.log(pred_bN)) ** 2
    rmse_log_b = torch.sqrt(torch.nanmean(rmse_log_bN, dim=1))

    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1)

    sq_rel_b = torch.nanmean((gt_bN - pred_bN) ** 2 / gt_bN, dim=1)

    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1)

    metrics_dict = {
                    "abs_diff": abs_diff_b,
                    "abs_rel": abs_rel_b,
                    "sq_rel": sq_rel_b,
                    "rmse": rmse_b,
                    "rmse_log": rmse_log_b,
                }
    metrics_dict.update(a_dict)

    return metrics_dict

def compute_metrics_for_sample(
    sample_path, 
    dataset_root, 
    sample_save_path, 
    debug_dump=False, 
    scene_name=None, 
    verbose=False, 
    clip_with_visibility=False
):

    # get separate visualizers for gt and pred, since the visualizer code uses the same internal variable for meshes.
    gt_visualizer = get_visualizer(sample_path, dataset_root, mode="gt", scene_name=scene_name, verbose=verbose)
    pred_visualizer = get_visualizer(sample_path, dataset_root, mode="pred", scene_name=scene_name, verbose=verbose)

    gt_depth_b1hw = torch.tensor(gt_visualizer.data['gt'])[None, None].cuda()
    pred_depth_b1hw = torch.tensor(pred_visualizer.data['pred'])[None, None].cuda()


    depth_metrics = compute_depth_metrics_batched(gt_depth_b1hw.flatten(1), pred_depth_b1hw.flatten(1), (gt_depth_b1hw > 0).flatten(1))

    # if debug_dump:
        # dump the volume
        # pcd_visibility_volume_save_path = Path(sample_save_path) / "pcd_visibility_volume.ply"
        # o3d.io.write_point_cloud(str(pcd_visibility_volume_save_path), visibility_volume.to_point_cloud(threshold=0.5, num_points=100000))
        
        # invK_b44 = torch.linalg.inv(torch.from_numpy(gt_visualizer.fix_camera_intrinsics(gt_visualizer.cameras['K_p'].copy(), [256,256])).cuda().clone().unsqueeze(0))
    
        # backprojector = BackprojectDepth(width=256, height=256).cuda()
        # points = backprojector(depth_b1hw, invK_b44)
        # points = world_T_cam_b44 @ points
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.squeeze().permute(1,0)[:, :3].cpu().numpy())
        # o3d.io.write_point_cloud(str(Path(sample_save_path) / "backproj_pers_mine.ply"), pcd)
        # gt_visualizer.get_pointcloud_depth_perspective_in_world_coordinates(path_to_save=Path(sample_save_path) / "backproj_pers.ply", is_save=True)
        
    
    return depth_metrics

def main(sample_paths, dataset_root, save_path, debug_dump=False, scene_name=None, verbose=False, clip_with_visibility=False):
    
    all_metrics = []
    # sample_paths = sample_paths[316:]
    for sample_path in tqdm(sample_paths):
        # create a folder for each sample
        sample_save_path = Path(save_path) / sample_path.split('/')[-1].split('.')[0]
        sample_save_path.mkdir(exist_ok=True, parents=True)
        
        sample_metrics = compute_metrics_for_sample(
            sample_path=sample_path, 
            dataset_root=dataset_root, 
            sample_save_path=sample_save_path, 
            debug_dump=debug_dump, 
            scene_name=scene_name, 
            verbose=verbose, 
            clip_with_visibility=clip_with_visibility,
        )
        
        all_metrics.append(sample_metrics)
    
    print(all_metrics)
    
    # aggregate metrics
    metrics = {}
    for metric_name in all_metrics[0].keys():
        metrics[metric_name] = torch.tensor([sample_metrics[metric_name] for sample_metrics in all_metrics]).mean().item()
    
    # add some extra info
    metrics['num_samples'] = len(all_metrics)
    metrics["sample_path"] = str(Path(sample_paths[0]).parent.resolve())
    
    # save metrics in a json file
    metrics_save_path = Path(save_path) / "depth_metrics.json"
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        


@click.command()
@click.option(
    "--dataset_path",
    type=str,
    default="/mnt/data_s/gunlu/data_gen/UrbanScene3D",
    help="Root directory of datasets",
)
@click.option(
    "--dataset_name",
    type=str,
    default="sf_test_1k_2",
    help="dataset name",
)
@click.option(
    "--predictions_path",
    type=str,
    default="/mnt/data_s/gunlu/data_gen/UrbanScene3D/sf_test_1k_2",
    help="Root directory of predictions",
)
@click.option(
    "--model_name",
    type=str,
    default="test231203_000446_v14_guide_gt_pt_rn_pt_vae_l1l2norm_max5.2_lr3",
    help="Model name",
)
@click.option(
    "--save_path",
    type=str,
    help="Save path.",
)
@click.option(
    "--debug_dump",
    is_flag=True,
)
@click.option(
    "--verbose",
    is_flag=True,
)
@click.option(
    "--clip_with_visibility",
    is_flag=True,
)
def cli(
    dataset_path: str, 
    dataset_name: str, 
    predictions_path: str, 
    model_name: str, 
    save_path: str, 
    debug_dump: bool=False, 
    verbose: bool=False, 
    clip_with_visibility: bool=False,
):

    # get paths of samples to evaluate. They should have "_gt.npy" in the name,
    # and there should be a corresponding prediction file with with "_pred.npy" in the name.
    sample_paths = sorted(glob.glob(os.path.join(predictions_path, '*.npy')))
    sample_paths = [sample_path for sample_path in sample_paths if "_gt.npy" in Path(sample_path).name]
    
    main(
        sample_paths,
        dataset_root=os.path.join(dataset_path, dataset_name),
        save_path=save_path,
        scene_name=dataset_name,
        debug_dump=debug_dump,
        verbose=verbose,
        clip_with_visibility=clip_with_visibility,
    )

if __name__ == '__main__':
    cli()